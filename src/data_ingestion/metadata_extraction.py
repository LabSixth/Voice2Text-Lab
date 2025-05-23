
import logging
import os
import polars as pl
import soundfile as sf
import dagster as dg
from src import global_configs as cf
from src.data_ingestion import web_download

# Get configurations
logger = logging.getLogger(__name__)
CONFIG = cf.PIPELINE_CONFIG["Data_Processing_Pipeline"]
RAW_DATA = cf.DATA_PATH.joinpath(CONFIG["Folder_Tree"]["Raw_Data"]).resolve()
METADATA = cf.DATA_PATH.joinpath(CONFIG["Folder_Tree"]["Metadata"]).resolve()


@dg.asset(
    deps=[web_download.unpack_move],
    kinds={"python", "json"}
)
def file_structure_gather() -> dict:
    """
    Gathers information about the file structure in a specified raw data directory.

    The function iterates through directories representing user IDs in a raw data
    directory. It further collects information about chapter directories contained
    within each user directory. This data is compiled into a metadata dictionary
    containing user IDs and their respective chapters.

    Returns:
        dict: A dictionary containing user IDs and their corresponding chapter data.
        The structure is:
            {
                "user_ids": list of str,
                "chapters": list of list of str
            }
    """

    # Get a list of folders from the raw data folder - representing individual User IDs
    user_ids = [p.name for p in RAW_DATA.iterdir() if p.is_dir()]

    # For each User ID, list all the chapters that the user has audio on
    logger.info("Starting process to extract users and chapters information.")
    user_chapters = []
    for user_id in user_ids:
        chapters_path = RAW_DATA.joinpath(user_id).resolve()
        chapters = [p.name for p in chapters_path.iterdir() if p.is_dir()]

        # Save the list of chapters into the running list
        user_chapters.append(chapters)

    # Create a dictionary to save the metadata
    metadata = {"user_ids": user_ids, "chapters": user_chapters}
    logger.info("Completed extraction pipeline and returning metadata related for next step processing.")
    return metadata


@dg.asset(
    ins={"user_meta": dg.AssetIn(key="file_structure_gather")},
    kinds={"python", "json", "polars"}
)
def metadata_gather(user_meta: dict) -> pl.DataFrame:
    """
    Gathers metadata for user recordings across multiple chapters and organizes it
    into a formatted dataframe. This function extracts recording lengths and relevant
    details from specific folders, processes them, and organizes the data in a
    structured format suitable for further analysis.

    Args:
        user_meta (dict): A dictionary containing metadata about users and chapters.
            It must include:
            - user_ids (list[str]): A list of user IDs.
            - chapters (list[list[str]]): A nested list where each inner list contains
              chapter IDs associated with the corresponding user in `user_ids`.

    Returns:
        pl.DataFrame: A Polars dataframe with the following columns:
            - user_id (int): ID of the user.
            - chapter_id (int): ID of the chapter.
            - recording_id (int): ID of the recording within a chapter.
            - recording_length (float): Duration (in seconds) of the recording.
            - recording_file (str): Name of the recording file.
    """

    # Get all the users and all the chapters
    all_chapters = user_meta["chapters"]

    # For each user, get the metadata of the recording for all chapters
    logger.info("Starting process to audio files information from downloaded data.")
    df = pl.DataFrame()
    for idx, user_id in enumerate(user_meta["user_ids"]):
        user_chapters = all_chapters[idx]
        chapter_df = pl.DataFrame()
        for chapter in user_chapters:
            # List all the recordings in the folder
            recording_path = RAW_DATA.joinpath(user_id, chapter).resolve()
            recordings = [p.name for p in recording_path.iterdir() if p.is_file() and p.suffix == '.flac']

            # For each of the recording, extract the length of the recording and save all relevant data into a dataframe
            recordings_df = pl.DataFrame()
            for recording in recordings:
                info = sf.info(recording_path.joinpath(recording).resolve())
                recordings_df = (
                    recordings_df
                    .vstack(pl.DataFrame({"recording_length": info.duration, "recording_file": recording}))
                )

            # Save the data into the larger dataframe
            chapter_df = chapter_df.vstack(recordings_df.with_columns(pl.lit(chapter).alias("chapter_id")))

        # Save the final dataframe into the user group
        df = df.vstack(chapter_df.with_columns(pl.lit(user_id).alias("user_id")))

    # Final formatting the dataframe before saving
    df = (
        df
        .with_columns(
            (
                pl.col("recording_file").str.split("-")
                .list.last()
                .str.split(".")
                .list.first()
            ).alias("recording_id")
        )
        .with_columns(
            pl.col("recording_id").cast(pl.Int64) + 1,
            pl.col(["user_id", "chapter_id"]).cast(pl.Int64)
        )
        .select("user_id", "chapter_id", "recording_id", "recording_length", "recording_file")
        .sort("user_id", "chapter_id", "recording_id")
        .with_row_index(name="id", offset=1)
    )
    logger.info("Completed process of extracting metadata of downloaded audio files.")

    return df


@dg.asset(
    ins={"df": dg.AssetIn(key="metadata_gather")},
    kinds={"python", "polars", "parquet"}
)
def save_metadata(df: pl.DataFrame) -> None:
    # Get the configurations of save file
    filename = CONFIG["Metadata_Configurations"]["Filename"]
    save_path = METADATA.joinpath(filename).resolve()

    # Make sure that the folder exists
    os.makedirs(METADATA, exist_ok=True)

    # Save the data using the configuration provided
    logger.info(f"Saving metadata into {CONFIG["Metadata_Configurations"]["Save_Format"]} file.")
    if CONFIG["Metadata_Configurations"]["Save_Format"] == "parquet":
        df.write_parquet(
            file=save_path,
            compression=CONFIG["Metadata_Configurations"]["Compression"],
            compression_level=CONFIG["Metadata_Configurations"]["Compression_Level"]
        )

    else:
        df.write_csv(save_path)
