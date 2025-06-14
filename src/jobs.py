
from dagster import define_asset_job

# Define Dagster jobs
run_download_pipeline = define_asset_job(
    name="run_download_pipeline",
    selection=[
        "download_data", "unpack_move", "clean_up",
        "file_structure_gather", "metadata_gather", "save_metadata",
        "speech_to_text_conversion", "save_transcriptions", "create_full_dataset"
    ]
)

run_modeling_pipeline = define_asset_job(
    name="run_modeling_pipeline",
    selection=[
        "data_sourcing", "t5_summarization", "save_summaries",
        "entity_recognition", "save_entities",
        "combine_data"
    ]
)
