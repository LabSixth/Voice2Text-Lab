
from dagster import load_assets_from_package_module, Definitions
from src import data_ingestion, ner_summarizations
from src import jobs

# Load all assets definitions
data_ingestion_assets = load_assets_from_package_module(package_module=data_ingestion)
modeling_assets = load_assets_from_package_module(package_module=ner_summarizations)
all_assets = [
    *data_ingestion_assets, *modeling_assets
]

defs = Definitions(
    assets=all_assets,
    jobs=[
        jobs.run_download_pipeline, jobs.run_modeling_pipeline
    ]
)
