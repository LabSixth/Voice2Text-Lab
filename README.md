# What are they saying? 🗣

This repository is a voice-to-text transcription application powered by modern speech recognition technologies.
This project allows users to convert spoken audio into readable text with high accuracy and minimal latency.

---

## Features

- 🎙️ Real-time or batch audio transcription
- 💬 Support for multiple languages and accents
- ⚡ Fast and scalable processing pipeline
- 🧠 Powered by state-of-the-art speech recognition models
- 🗃️ Easy integration with audio input/output sources

## Quick Start ⚙️

### Prerequisite

1. Python 3.11 or newer
2. `uv` package manager installed (recommended). Follow quick [installation](https://docs.astral.sh/uv/getting-started/installation/) guide.

### Getting Started

Clone this repository to local machine.

```shell
git clone git@github.com:LabSixth/text-analytics.git
cd text-analytics
```

Using your choice of package manager, create a virtual environment and install the dependencies for this project.

*Virtual Environment Using `uv`*

`uv sync` creates the virtual environment and install the dependencies for this project at the same time.

```shell
uv sync
source .venv/bin/activate
```

*Virtual Environment Using `pip`*

Alternatively, `pip` can be used to create the virtual environment and install the dependencies for this project.

```shell
# For Linux and MacOS
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# For Windows
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Pre-built Pipeline 🚀

This repository has a pre-built pipeline to run an end-to-end speech-to-text conversion, along with summarization and Name
Entities Recognition (NER) using Google's T5 (for summarization) and GliNER (for NER). Data is pulled from
[LibriSpeech](https://www.openslr.org/12) `dev-clean.tar.gz`.

To change the dataset pulled from LibriSpeech, change the `Source_Download` section in `configs/pipeline_configs.yaml`.
End-to-end pipeline uses [Dagster](https://dagster.io/) orchestration tool.

> [!NOTE]
> Full pipeline takes about 30 minutes to run using the dataset above.

To run the pipeline to download, clean, and extract textual data from recording, run the following command in terminal.

```shell
dagster dagster job execute -m src -j run_download_pipeline
```

To run the summarization and NER pipeline, run the following command in terminal.

```shell
dagster job execute -m src -j run_modeling_pipeline
```

---

## Streamlit Application 🌐

This project comes with a Streamlit web application, allowing users to upload either audio recording or a song of choice, to
perform summarization and NER.

To start up the Streamlit application, run the following command in terminal.

```shell
streamlit run app.py
```

---

## Contributions

Thanks to the following people for their contributions:

- [Yuxuan Huang](https://github.com/Eleanorhhhyxz)
- [Moyi Li](https://github.com/Moyi-Li)
- [Mingsha Mo](https://github.com/monicamomingsha)
- [Yee Jun Ow](https://github.com/YeeJunOw19)
- [Tianyi Zhang](https://github.com/th3ch103)

