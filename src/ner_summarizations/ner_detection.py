
import polars as pl
import os
import logging
import dagster as dg
from gliner import GLiNER
from src import global_configs as cf
from src.ner_summarizations import text_summarization

logger = logging.getLogger(__name__)
CONFIGS = cf.PIPELINE_CONFIG["Summarization_Named_Entity_Recognition"]


@dg.asset(
    ins={"data": dg.AssetIn(key="data_sourcing")},
    kinds={"python", "polars", "huggingface"}
)
def entity_recognition(data: dict) -> pl.DataFrame:
    """
    Processes a given dataset to perform Named Entity Recognition (NER) using a pretrained
    GLiNER model and returns a DataFrame containing extracted entities.

    This function leverages a GLiNER model to identify and extract entities from text
    transcripts provided in the input data. The extracted entities are then combined
    with the original data as a new column to create the final DataFrame.

    Args:
        data (dict): A dictionary containing the input data. It must include:
            - "transcripts" (list of str): A list of text strings for which to perform NER.
            - "df" (pl.DataFrame): A Polars DataFrame with columns 'user_id' and 'chapter_id'
              used for retaining the mapping of the extracted entities.

    Returns:
        pl.DataFrame: A Polars DataFrame with the original 'user_id' and 'chapter_id' columns
        from the input data and an additional column 'extracted_entities' containing the
        entities extracted by the GLiNER model for each corresponding transcript.
    """

    # Get configurations for NER
    model_ident = CONFIGS["Named_Entity_Models"]["Gliner_Identifier"]
    model_configs = cf.MODELS_CONFIG[model_ident]

    # Create an instance of Gliner for NER
    model = GLiNER.from_pretrained(
        pretrained_model_name_or_path=model_configs["Model_Name"],
        load_tokenizer=True,
        max_length=model_configs["Maximum_Length"]
    ).to("cuda" if cf.DEVICE == "auto" else cf.DEVICE).eval()
    labels = model_configs["Labels"]

    # For each of the recording, extract NER
    text_entities = []
    for text in data["transcripts"]:
        entities = model.predict_entities(text, labels)
        text_entities.append(entities)

    # Combine the data into dataframe and proceed to saving next
    df = (
        data["df"]
        .hstack(pl.DataFrame({"extracted_entities": text_entities}))
        .select("user_id", "chapter_id", "extracted_entities")
    )

    return df


@dg.asset(
    ins={"df": dg.AssetIn(key="entity_recognition")},
    kinds={"python", "polars", "parquet"}
)
def save_entities(df: pl.DataFrame) -> None:
    """
    Save processed entities to a specified output file based on configurations.

    This function is responsible for saving the processed entity recognition data
    to a file in the configured format (e.g., Parquet or CSV) within a specified
    output directory. It ensures the directory structure exists and applies the
    necessary compression settings if using Parquet format.

    Args:
        df: The processed entity recognition data represented as a polars DataFrame.
    """

    # Get the configurations of save file
    folder_name = CONFIGS["Folder_Tree"]["Named_Entity_Outputs"]["Folder_Name"]
    filename = CONFIGS["Folder_Tree"]["Named_Entity_Outputs"]["Filename"]
    save_path = cf.DATA_PATH.joinpath(folder_name, filename).resolve()

    # Make sure that the folder exists
    os.makedirs(cf.DATA_PATH.joinpath(folder_name).resolve(), exist_ok=True)

    # Save the data using the configuration provided
    logger.info(f"Saving metadata into {save_path} file.")
    if CONFIGS["Folder_Tree"]["Named_Entity_Outputs"]["Save_Format"] == "parquet":
        df.write_parquet(
            file=save_path,
            compression=CONFIGS["Folder_Tree"]["Named_Entity_Outputs"]["Compression"],
            compression_level=CONFIGS["Folder_Tree"]["Named_Entity_Outputs"]["Compression_Level"]
        )

    else:
        df.write_csv(save_path)
