
import polars as pl
import os
import dagster as dg
from pathlib import Path
from src import global_configs as cf
from src.data_ingestion.text_extraction import save_transcriptions
from src.data_ingestion.metadata_extraction import save_metadata

CONFIGS = cf.PIPELINE_CONFIG["Summarization_Named_Entity_Recognition"]


def _read_data(filepath: Path) -> pl.DataFrame:
    """
    Reads a data file and returns it as a polars DataFrame. Determines the file type
    based on the file extension and reads the data accordingly. Supports `.parquet`
    and `.csv` file formats.

    Args:
        filepath (Path): Path object representing the file to be read.

    Returns:
        pl.DataFrame: A polars DataFrame containing the data from the file.
    """

    # Get the file extension of the given file
    file_ext = filepath.__str__().split(".")[-1]

    if file_ext == "parquet":
        df = pl.read_parquet(filepath)
    else:
        df = pl.read_csv(filepath)

    return df


@dg.asset(
    deps=[save_transcriptions, save_metadata],
    kinds={"python", "polars", "parquet"}
)
def create_full_dataset() -> None:
    """
    Combines metadata and transcription data into a single dataset and saves it in
    the specified format and location. The function utilizes configurations for paths
    and file details, ensures the output directory exists, and creates the dataset by
    joining and processing the provided metadata and transcription files.
    """

    # Get configurations to for this task
    save_folder = cf.DATA_PATH.joinpath(CONFIGS["Folder_Tree"]["Combined_Data"]["Folder_Name"]).resolve()
    os.makedirs(save_folder, exist_ok=True)

    meta_df = (
        cf.DATA_PATH
        .joinpath(
            CONFIGS["Folder_Tree"]["Cleaned_Data"]["Folder_Nam"],
            CONFIGS["Folder_Tree"]["Cleaned_Data"]["Metadata_File"]
        )
        .resolve()
    )

    transcripts_df = (
        cf.DATA_PATH
        .joinpath(
            CONFIGS["Folder_Tree"]["Cleaned_Data"]["Folder_Nam"],
            CONFIGS["Folder_Tree"]["Cleaned_Data"]["Transcription_File"]
        )
        .resolve()
    )

    # Combine the metadata and transcriptions together into one dataset
    df = (
        _read_data(transcripts_df)
        .join(_read_data(meta_df), on="id", how="left")
        .sort("user_id", "chapter_id", "id")
        .group_by("user_id", "chapter_id", maintain_order=True)
        .agg(
            pl.col("recording_transcriptions"),
            pl.col("recording_length").sum().alias("recording_length")
        )
        .with_columns(
            pl.col("recording_transcriptions").list.join(" ").alias("recording_transcriptions")
        )
    )

    # Save the data
    save_file = save_folder.joinpath(CONFIGS["Folder_Tree"]["Combined_Data"]["Filename"]).resolve()
    if CONFIGS["Folder_Tree"]["Combined_Data"]["Save_Format"] == "parquet":
        df.write_parquet(
            file=save_file,
            compression=CONFIGS["Folder_Tree"]["Combined_Data"]["Compression"],
            compression_level=CONFIGS["Folder_Tree"]["Combined_Data"]["Compression_Level"]
        )

    else:
        df.write_csv(save_file)
