
from dagster import define_asset_job

# Define Dagster jobs
run_download_pipeline = define_asset_job(
    name="run_download_pipeline",
    selection=["download_data", "unpack_move", "clean_up"]
)
