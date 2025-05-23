
import polars as pl
import os
import logging
import dagster as dg
from tqdm import tqdm
from src import global_configs as cf
from tools.models import google_flan
from src.data_ingestion import text_preprocessing

logger = logging.getLogger(__name__)
CONFIGS = cf.PIPELINE_CONFIG["Summarization_Named_Entity_Recognition"]


@dg.asset(
    deps=[text_preprocessing.create_full_dataset],
    kinds={"python", "parquet", "polars"}
)
def data_sourcing() -> dict:
    """
    Fetches and processes data from a specified folder and file, returning both the data
    as a DataFrame and the transcription column.

    The function utilizes the configuration settings to locate the folder and file paths
    where the data is stored. It handles data files in either Parquet or CSV format and
    extracts a specified column of transcriptions for further use.

    Returns:
        dict: A dictionary containing the following keys:
            - "df": The DataFrame containing the complete data read from the file.
            - "transcripts": The transcription column extracted from the DataFrame.
    """

    # Get configurations of the data being used
    data_folder = cf.DATA_PATH.joinpath(CONFIGS["Folder_Tree"]["Combined_Data"]["Folder_Name"]).resolve()
    data_file = data_folder.joinpath(CONFIGS["Folder_Tree"]["Combined_Data"]["Filename"]).resolve()
    file_ext = data_file.__str__().split(".")[-1]

    # Read in data and split the data
    df = pl.read_parquet(data_file) if file_ext == "parquet" else pl.read_csv(data_file)
    transcripts = df["recording_transcriptions"]
    
    return {"df": df, "transcripts": transcripts}


@dg.asset(
    ins={"data": dg.AssetIn(key="data_sourcing")},
    kinds={"python", "huggingface", "google"}
)
def t5_summarization(data: dict) -> pl.DataFrame:
    """
    Generates text summaries of varying lengths using the T5 summarization model, then combines the
    summaries with the original data into a new DataFrame.

    This function utilizes a pre-trained T5 model to create summaries of input text from the provided
    data dictionary. The summaries are generated for three distinct configurations: short, medium, and
    large output lengths. Each configuration specifies minimum and maximum token limits for the
    summaries. The resulting summaries are combined with the original data and returned as a new
    Polars DataFrame.

    Args:
        data (dict): A dictionary containing the input data. It should have the following keys:
            - "transcripts" (list of str): A list of text strings to summarize.
            - "df" (pl.DataFrame): A Polars DataFrame containing the original dataset to be combined
              with the generated summaries.

    Returns:
        pl.DataFrame: A new Polars DataFrame that includes the original columns "user_id" and
        "chapter_id" alongside the generated summaries under columns "t5_short", "t5_medium", and
        "t5_large".
    """

    # Get all configurations
    model_ident = CONFIGS["Summarization_Models"]["T5_Model_Identifier"]
    model_configs = cf.MODELS_CONFIG[model_ident]
    token_length = model_configs["Maximum_Token_Generation"]

    summarization_configs = [
        (token_length["Short_Output"]["Minimum_Length"], token_length["Short_Output"]["Maximum_Length"]),
        (token_length["Medium_Output"]["Minimum_Length"], token_length["Medium_Output"]["Maximum_Length"]),
        (token_length["Large_Output"]["Minimum_Length"], token_length["Large_Output"]["Maximum_Length"])
    ]

    # Instantiate a T5 model
    model = google_flan.GoogleFlanT5(
        model_name=model_configs["Model_Name"],
        device=cf.DEVICE,
        token_required=model_configs["Hugging_Face_Token"],
        token=None
    )

    # For each of the summarization length, create summarizations for all text
    summaries = []
    for config in summarization_configs:
        min_length, max_length = config
        summarizations = []

        for text in tqdm(data["transcripts"], desc="Summarizing transcripts"):
            summary = model.inference(
                input_text=f"summarize: {text}",
                min_length=min_length,
                max_length=max_length
            )
            summarizations.append(summary)

        summaries.append(summarizations)

    # Move the summaries back into a dataframe and join it with the original dataframe
    df = (
        data["df"]
        .hstack(pl.DataFrame({"t5_short": summaries[0], "t5_medium": summaries[1], "t5_large": summaries[2]}))
        .select("user_id", "chapter_id", "t5_short", "t5_medium", "t5_large")
    )

    return df


@dg.asset(
    ins={"df": dg.AssetIn(key="t5_summarization")},
    kinds={"python", "polars", "parquet"}
)
def save_summaries(df: pl.DataFrame) -> None:
    """
    Saves the provided DataFrame to a specified location in a specified format. The function retrieves
    the output folder name, file name, and save format (parquet or CSV) from configuration settings.
    If the folder does not exist, it creates the folder before saving the data. For parquet files, it
    uses provided compression settings.

    Args:
        df (pl.DataFrame): The DataFrame that contains the data to be saved. The DataFrame will be
            saved based on the file format and configurations specified in the CONFIGS object.
    """

    # Get the configurations of save file
    folder_name = CONFIGS["Folder_Tree"]["Summarization_Outputs"]["Folder_Name"]
    filename = CONFIGS["Folder_Tree"]["Summarization_Outputs"]["Filename"]
    save_path = cf.DATA_PATH.joinpath(folder_name, filename).resolve()

    # Make sure that the folder exists
    os.makedirs(folder_name, exist_ok=True)

    # Save the data using the configuration provided
    logger.info(f"Saving metadata into {save_path} file.")
    if CONFIGS["Folder_Tree"]["Summarization_Outputs"]["Save_Format"] == "parquet":
        df.write_parquet(
            file=save_path,
            compression=CONFIGS["Folder_Tree"]["Summarization_Outputs"]["Compression"],
            compression_level=CONFIGS["Folder_Tree"]["Summarization_Outputs"]["Compression_Level"]
        )

    else:
        df.write_csv(save_path)
