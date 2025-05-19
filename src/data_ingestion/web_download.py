
import httpx
import logging
import shutil
import os
import tarfile
import dagster as dg
from pathlib import Path
from src import global_configs as cf

logger = logging.getLogger(__name__)

CONFIG = cf.PIPELINE_CONFIG["Data_Processing_Pipeline"]


@dg.asset(kinds={"python"})
def download_data() -> None:
    """
    Downloads data from a specified source URL to a temporary folder.

    This function retrieves configuration settings for the source URL and the temporary
    download folder location, ensures that the temporary folder does not already exist,
    and then downloads the data from the source URL. The downloaded data is stored in
    a specified temporary folder as a file. Download progress is logged, and an
    exception is automatically raised for any HTTP errors encountered during the
    download process.

    Raises:
        HTTPError: If the HTTP request encounters an error during download.
    """

    # Get the configurations of temporary download location
    data_url = CONFIG["Source_Download"]
    temp_download_folder = CONFIG["Folder_Tree"]["Temp_Zip"]

    # Make sure that the temporary folder does not exist
    if os.path.exists(temp_download_folder):
        os.remove(temp_download_folder)

    # Download the file
    logger.info(f"Downloading data from {data_url}...")
    with httpx.stream("GET", data_url, follow_redirects=True) as response:
        response.raise_for_status()
        with open(temp_download_folder, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    logger.info(f"Data downloaded successfully!")


@dg.asset(kinds={"python"}, deps=[download_data])
def unpack_move() -> None:
    """
    Unpacks and moves data from a temporary zipped location to a landing zone.

    This function handles the configuration-based unpacking of a tar.gz file, its temporary
    processing, and the final transfer of unzipped files to a designated landing zone. It
    ensures cleanup of temporary directories and makes sure the directories exist as
    required during the process.
    """

    # Get all the configurations needed to unpack and move the data
    temp_download_folder = CONFIG["Folder_Tree"]["Temp_Zip"]
    temp_unpack_folder = CONFIG["Folder_Tree"]["Temp_Unzip"]
    unzipped_data_loc = CONFIG["Corpus_Structure"]
    landing_zone = cf.DATA_PATH.joinpath(CONFIG["Folder_Tree"]["Raw_Data"]).resolve()

    # Make sure that the landing zone exists
    if os.path.exists(landing_zone):
        shutil.rmtree(landing_zone)
    os.makedirs(landing_zone, exist_ok=True)

    # Remove the temporary folder if it exists
    if os.path.exists(temp_unpack_folder):
        shutil.rmtree(temp_unpack_folder)

    # Unzip the tar.gz file
    logger.info(f"Unzipping data...")
    with tarfile.open(temp_download_folder, "r:gz") as tar:
        tar.extractall(temp_unpack_folder)
    logger.info(f"Data unzipped successfully!")

    # Move all the files from the unzipped folder to landing zone
    data = Path(temp_unpack_folder).joinpath(unzipped_data_loc).resolve()
    for item in data.iterdir():
        dst_path = landing_zone.joinpath(item.name).resolve()
        shutil.move(item, dst_path)


@dg.asset(kinds={"python"}, deps=[download_data, unpack_move])
def clean_up() -> None:
    """
    Deletes temporary artifacts generated during the process to ensure that no
    residual data remains. This function specifically removes files and
    directories that are stored in the temporary download and unpack folders.
    """

    # Delete the temporary artifacts
    temp_download_folder = CONFIG["Folder_Tree"]["Temp_Zip"]
    temp_unpack_folder = CONFIG["Folder_Tree"]["Temp_Unzip"]
    if os.path.exists(temp_download_folder):
        os.remove(temp_download_folder)
    if os.path.exists(temp_unpack_folder):
        shutil.rmtree(temp_unpack_folder)
