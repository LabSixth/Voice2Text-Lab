
import polars as pl
import os
import dagster as dg
from src import global_configs as cf

CONFIGS = cf.PIPELINE_CONFIG["Summarization_Named_Entity_Recognition"]


@dg.asset(
    ins={"df_entities": dg.AssetIn(key="entity_recognition"), "df_summarized": dg.AssetIn(key="t5_summarization")},
    kinds={"python", "polars", "parquet"}
)
def combine_data(df_entities: pl.DataFrame, df_summarized: pl.DataFrame) -> None:
    """
    Combines entity data and summarized text data into a single representation,
    cleans the summarized text, and saves the resultant combined data to a file.
    The function processes two dataframes:
        1. An exploded and unnested entity dataframe with pivoted and renamed fields.
        2. A cleaned summarized dataframe with redundant tags and whitespace removed.
    The resultant dataframe is saved in either Parquet or CSV format based on
    configurations.

    Args:
        df_entities: A Polars DataFrame containing entity information with nested
            and unnested `extracted_entities` attributes. It includes fields for
            user identifiers, chapters, labels, text, and scores associated with
            extracted entities.
        df_summarized: A Polars DataFrame containing summarized textual data with
            columns such as `t5_short`, `t5_medium`, and `t5_large`, which will be
            cleaned, removing undesired tags and whitespace.
    """

    # Get configurations
    folder_path = cf.DATA_PATH.joinpath(CONFIGS["Folder_Tree"]["Combined_Output"]["Folder_Name"]).resolve()
    file_path = folder_path.joinpath(CONFIGS["Folder_Tree"]["Combined_Output"]["Filename"]).resolve()
    os.makedirs(folder_path, exist_ok=True)

    # Unnest the entity dataframe
    df_entities = (
        df_entities
        .explode("extracted_entities")
        .unnest("extracted_entities")
        .group_by("user_id", "chapter_id", "label", maintain_order=True)
        .agg(pl.col("text"), pl.col("score"))
        .filter(pl.col("label").is_not_null())
        .pivot(on="label", index=["user_id", "chapter_id"], values=["text", "score"])
        .rename(
            {
                "text_Persons": "persons_text", "text_Location": "location_text", "text_Organization": "org_text",
                "score_Persons": "persons_score", "score_Location": "location_score", "score_Organization": "org_score"
            }
        )
    )

    # Clean the summarized text
    df_summarized = (
        df_summarized
        .with_columns(pl.col(["t5_short", "t5_medium", "t5_large"]).str.replace_all("<pad>", ""))
        .with_columns(pl.col(["t5_short", "t5_medium", "t5_large"]).str.replace_all("</s>", ""))
        .with_columns(pl.col(["t5_short", "t5_medium", "t5_large"]).str.replace_all("summary:", ""))
        .with_columns(pl.col(["t5_short", "t5_medium", "t5_large"]).str.strip_chars(" "))
    )

    # Combine the dataframes and save the data
    df = (
        df_summarized
        .join(df_entities, on=["user_id", "chapter_id"], how="left")
        .with_row_index(name="id", offset=1)
    )

    if CONFIGS["Folder_Tree"]["Combined_Output"]["Save_Format"] == "parquet":
        df.write_parquet(
            file=file_path,
            compression=CONFIGS["Folder_Tree"]["Combined_Output"]["Compression"],
            compression_level=CONFIGS["Folder_Tree"]["Combined_Output"]["Compression_Level"]
        )

    else:
        df.write_csv(file_path)