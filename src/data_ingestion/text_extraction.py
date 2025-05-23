
import polars as pl
import logging
import os
from pathlib import Path
from tqdm import tqdm
import dagster as dg
from tools.models import whisper_ai
from src import global_configs as cf
from src.data_ingestion.metadata_extraction import METADATA

# Get configurations for the run
logger = logging.getLogger(__name__)
TASK_CONFIG = cf.PIPELINE_CONFIG["Data_Processing_Pipeline"]
MODEL_CONFIG = cf.MODELS_CONFIG[TASK_CONFIG["Model_Identifier"]]
ROOT_PATH = cf.DATA_PATH.joinpath(TASK_CONFIG["Folder_Tree"]["Raw_Data"]).resolve()


@dg.asset(
    ins={"df": dg.AssetIn(key="metadata_gather")},
    kinds={"python", "polars", "huggingface"}
)
def speech_to_text_conversion(df: pl.DataFrame) -> pl.DataFrame:
    # Get processing configurations
    batch_size = TASK_CONFIG["Maximum_Batch_Size"]

    # Create an instance of Whisper model
    whisper_model = whisper_ai.WhisperAI(
        model_name="openai/whisper-large-v3",
        model_task="automatic-speech-recognition",
        token_required=False,
        token=None,
        device=cf.DEVICE
    )

    # From the dataframe get a list of files to be processed
    file_lists = (
        df
        .with_columns(
            pl.concat_str(
                [pl.col("user_id"), pl.col("chapter_id"), pl.col("recording_file")],
                separator="/"
            ).alias("file_path")
        )
        .select("file_path")
        .to_series()
        .to_list()
    )
    processing_list = [Path(ROOT_PATH).joinpath(x).resolve().__str__() for x in file_lists]

    # Perform batch inferencing on all the audio files
    audio_outputs = []
    for i in tqdm(range(0, len(processing_list), batch_size), desc="Transcribing audio batch"):
        batch = processing_list[i: i + batch_size]
        logger.info(f"\nRunning batch inference with batch size {len(batch)}.")

        model_output = whisper_model.inference(
            audio_files=batch,
            max_new_tokens=MODEL_CONFIG["Maximum_Token_Generation"],
            language=MODEL_CONFIG["Language_Selection"]
        )
        logger.info(f"\nCompleted batch inference with batch size {len(model_output)}.")

        if len(batch) != len(model_output):
            raise RuntimeError(f"\nInput batch is not the same size as output batch. {len(batch)} != {len(model_output)}")

        # Extract the actual text from each output and append it to the running list
        audio_outputs.extend([x["text"].strip() for x in model_output])

    # Save the transcription back into the dataframe
    transcription_df = (
        df
        .hstack(pl.DataFrame({"recording_transcriptions": audio_outputs}))
        .select("id", "recording_transcriptions")
    )

    return transcription_df


@dg.asset(
    ins={"df": dg.AssetIn(key="speech_to_text_conversion")},
    kinds={"python", "polars", "parquet"}
)
def save_transcriptions(df: pl.DataFrame) -> None:
    # Get the configurations of save file
    filename = TASK_CONFIG["Transcriptions_Configurations"]["Filename"]
    save_path = METADATA.joinpath(filename).resolve()

    # Make sure that the folder exists
    os.makedirs(METADATA, exist_ok=True)

    # Save the data using the configuration provided
    logger.info(f"Saving metadata into {TASK_CONFIG["Transcriptions_Configurations"]["Save_Format"]} file.")
    if TASK_CONFIG["Transcriptions_Configurations"]["Save_Format"] == "parquet":
        df.write_parquet(
            file=save_path,
            compression=TASK_CONFIG["Transcriptions_Configurations"]["Compression"],
            compression_level=TASK_CONFIG["Transcriptions_Configurations"]["Compression_Level"]
        )

    else:
        df.write_csv(save_path)
