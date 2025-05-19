
from dagster import load_assets_from_package_module, Definitions
from src import data_ingestion
from src import jobs

# Load all assets definitions
data_ingestion_assets = load_assets_from_package_module(package_module=data_ingestion)
all_assets = [
    *data_ingestion_assets,
]

defs = Definitions(
    assets=all_assets,
    jobs=[
        jobs.run_download_pipeline
    ]
)
