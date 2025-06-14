
import polars as pl
from src import global_configs as cf

CONFIGS = cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]


def extract_precomputed(id: int) -> dict:
    """
    Extracts pre-computed metadata for a specific identifier from stored Parquet files.

    This function locates and reads a pre-defined dataset to extract all metadata
    for the given 'id'. The result is returned as a dictionary containing the
    metadata entries without the 'id' key.

    Args:
        id (int): The identifier for which the metadata needs to be extracted. This
            should match the 'id' column in the pre-computed dataset.

    Returns:
        dict: A dictionary containing the extracted metadata for the given 'id'.
    """

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
    """
    Extracts original data for a specific user and chapter from pre-computed data.

    This function retrieves data associated with a given `user_id` and `chapter_id`
    from a pre-computed dataset file. The data is returned as a dictionary, excluding
    the user and chapter identifiers from the returned information.

    Args:
        user_id (int): The unique identifier for the user whose data is being requested.
        chapter_id (int): The unique identifier for the specific chapter whose data
            is being requested.

    Returns:
        dict: A dictionary containing the extracted data for the specified user
            and chapter, excluding user and chapter identifiers.
    """

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
    """
    Extracts and combines pre-computed and original data into a structured JSON object.

    The function retrieves pre-computed data corresponding to the provided `id` from both T5
    and Gliner. Using the retrieved `user_id` and `chapter_id`, it further extracts the
    original text and combines it with the pre-computed data into a singular JSON object
    for downstream processing or storage.

    Args:
        id: An integer representing the unique identifier for the data extraction.

    Returns:
        A dictionary containing the combined pre-computed and original data.
    """

    # Extract the pre-computed data from T5 and Gliner
    precomputed_data = extract_precomputed(id)

    # Using the extracted user_id and chapter_id, extract the original text
    user_id = precomputed_data["user_id"][0]
    chapter_id = precomputed_data["chapter_id"][0]
    original_data = extract_original(user_id, chapter_id)

    # Format the data and return the formatted JSON object
    combined_data = {**precomputed_data, **original_data}
    return combined_data
