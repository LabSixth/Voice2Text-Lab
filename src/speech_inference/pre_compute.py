
import polars as pl
from src import global_configs as cf

CONFIGS = cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]


def extract_precomputed(id: int) -> dict:
    # Get the file with the pre-computed data
    file_path = cf.DATA_PATH.joinpath(CONFIGS["Speech_Modeled_Data"]).resolve()

    # Extract the data requested into a dictionary
    id_df = (
        pl.scan_parquet(file_path)
        .filter(pl.col("id") == id)
        .select(pl.exclude("id"))
    )
    metadata = id_df.collect().to_dict(as_series=False)

    return metadata


def extract_original(user_id: int, chapter_id: int) -> dict:
    # Get the file with the pre-computed data
    file_path = cf.DATA_PATH.joinpath(CONFIGS["Speech_Original_Data"]).resolve()

    # Extract the data requested into a dictionary
    id_df = (
        pl.scan_parquet(file_path)
        .filter((pl.col("user_id") == user_id) & (pl.col("chapter_id") == chapter_id))
        .select(pl.exclude("user_id", "chapter_id"))
    )
    metadata = id_df.collect().to_dict(as_series=False)

    return metadata


def main_extraction(id: int) -> dict:
    # Extract the pre-computed data from T5 and Gliner
    precomputed_data = extract_precomputed(id)

    # Using the extracted user_id and chapter_id, extract the original text
    user_id = precomputed_data["user_id"][0]
    chapter_id = precomputed_data["chapter_id"][0]
    original_data = extract_original(user_id, chapter_id)

    # Format the data and return the formatted JSON object
    combined_data = {**precomputed_data, **original_data}
    return combined_data
