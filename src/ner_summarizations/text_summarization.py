
import polars as pl
from pathlib import Path
from src import global_configs as cf
from tools.models import google_flan

CONFIGS = cf.PIPELINE_CONFIG["Summarization_Named_Entity_Recognition"]


def data_sourcing() -> dict:
    # Get configurations of the data being used
    data_folder = cf.DATA_PATH.joinpath(CONFIGS["Folder_Tree"]["Combined_Data"]["Folder_Name"]).resolve()
    data_file = data_folder.joinpath(CONFIGS["Folder_Tree"]["Combined_Data"]["Filename"]).resolve()
    file_ext = data_file.__str__().split(".")[-1]

    # Read in data and split the data
    df = pl.read_parquet(data_file) if file_ext == "parquet" else pl.read_csv(data_file)
    transcripts = df["recording_transcriptions"]
    
    return {"df": df, "transcripts": transcripts}


def t5_summarization(data: dict) -> pl.DataFrame:
    # Get all configurations
    model_ident = CONFIGS["Summarization_Models"]["T5_Model_Identifier"]
    model_configs = cf.MODELS_CONFIG[model_ident]
    token_length = model_configs["Maximum_Token_Generation"]

    summarization_configs = [
        (token_length["Short_Output"]["Minimum_Length"], token_length["Short_Output"]["Maximum_Length"]),
        (token_length["Medium_Output"]["Minimum_Length"], token_length["Medium_Output"]["Maximum_Length"]),
        (token_length["Large_Output"]["Minimum_Length"], token_length["Large_Output"]["Maximum_Length"])
    ]
    print(summarization_configs)

    # Instantiate a T5 model
    model = google_flan.GoogleFlanT5(
        model_name=model_configs["Model_Name"],
        device=cf.DEVICE,
        token_required=model_configs["Hugging_Face_Token"],
        token=None
    )


test_data = data_sourcing()
t5_summarization(test_data)


