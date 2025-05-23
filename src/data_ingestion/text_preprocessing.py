
import polars as pl
import os
import dagster as dg
from src import global_configs as cf

CONFIGS = cf.PIPELINE_CONFIG["Summarization_Named_Entity_Recognition"]


@dg.asset(
    ins={
        "meta_df": dg.AssetIn(key="metadata_gather"),
        "transcripts_df": dg.AssetIn(key="speech_to_text_conversion")
    },
    kinds={"python", "polars", "parquet"}
)
def create_full_dataset(meta_df: pl.DataFrame, transcripts_df: pl.DataFrame) -> None:
    """
    Combines metadata and transcriptions into a single dataset, processes them by grouping
    and aggregating specific columns, and saves the resulting dataset to a specified
    directory in either Parquet or CSV format. The dataset is ordered based on user, chapter,
    and recording IDs.

    Args:
        meta_df (pl.DataFrame): A DataFrame containing metadata about the recordings.
        transcripts_df (pl.DataFrame): A DataFrame containing transcriptions of the recordings.
    """

    # Get configurations to for this task
    save_folder = cf.DATA_PATH.joinpath(CONFIGS["Folder_Tree"]["Combined_Data"]["Folder_Name"]).resolve()
    os.makedirs(save_folder, exist_ok=True)

    # Combine the metadata and transcriptions together into one dataset
    df = (
        transcripts_df
        .join(meta_df, on="id", how="left")
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
