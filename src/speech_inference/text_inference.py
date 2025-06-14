
import os
import shutil
import streamlit as st
from gliner import GLiNER
from streamlit.runtime.uploaded_file_manager import UploadedFile
from src import global_configs as cf
from tools.models import facebook_bart, microsoft_phi, whisper_ai, google_flan
from tools.utils import json_utils, streamlit_utils


@st.cache_data(ttl=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Object_TTL"])
def bart_inference(text: str, model_ident: str) -> str:
    """
    Performs text summarization using the BART (Bidirectional and Auto-Regressive Transformer)
    model. The function fetches the model configuration and initializes the BART model with
    proper settings. The input text is processed to generate a summary using the specified
    model.

    Args:
        text (str): The input text to be summarized.
        model_ident (str): Identifier for the BART model configuration to be used.

    Returns:
        str: The summarized text generated by the BART model.
    """

    # Get BART model configurations
    model_configs = cf.MODELS_CONFIG[model_ident]

    # Create an instance of BART model
    model = facebook_bart.FacebookBart(
        model_name=model_configs["Model_Name"],
        device=cf.DEVICE,
        task=model_configs["Model_Task"],
        token_required=model_configs["Hugging_Face_Token"],
        token=None
    )
    summary_output = model.inference(
        input_text=text,
        min_length=model_configs["Minimum_Length"],
        max_length=model_configs["Maximum_Length"]
    )

    return summary_output


@st.cache_data(ttl=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Object_TTL"])
def phi4_inference(text: str, model_ident: str, system_prompt: str, user_prompt: str) -> str:
    """
    Runs inference using the Phi4 model based on provided text input, model identifier,
    and specific prompt files for system and user interactions. The function reads
    predefined prompts, configures the model settings, and returns a summarization
    generated by the Phi4 model.

    Args:
        text: The user-provided input text for which the model needs to perform the inference.
        model_ident: Identifier for the model, used to fetch relevant configurations.
        system_prompt: Path to the file containing the system prompt, relative to a predefined
            prompts directory.
        user_prompt: Path to the file containing the user prompt, relative to a predefined
            prompts directory.

    Returns:
        str: The inference output (e.g., summary) generated by the Phi4 model.
    """

    # Get configurations for Phi4 and prompts
    model_configs = cf.MODELS_CONFIG[model_ident]
    system_prompt = cf.PROMPTS_PATH.joinpath(system_prompt).resolve()
    user_prompt = cf.PROMPTS_PATH.joinpath(user_prompt).resolve()

    # Read in system prompt and user prompt
    with open(system_prompt, "r", encoding="utf-8") as f:
        system_prompt_text = f.read()
    with open(user_prompt, "r", encoding="utf-8") as f:
        user_prompt_text = f.read()

    user_prompt_text = f"{user_prompt_text}\n{text}"

    # Instantiate Phi4 model
    model = microsoft_phi.Phi4Instruct(
        model_name=model_configs["Model_Name"],
        model_task=model_configs["Model_Task"],
        token_required=model_configs["Hugging_Face_Token"],
        token=None
    )

    # Generate summary
    summary_output = model.inference(
        system_prompt=system_prompt_text,
        user_prompt=user_prompt_text,
        max_new_tokens=model_configs["Maximum_New_Token"],
        temperature=model_configs["Temperature"],
        top_p=model_configs["Top_P"]
    )

    return summary_output


@st.cache_data(ttl=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Object_TTL"])
def full_inference_pipeline(
    file: UploadedFile, model_selection: str, temp_dir: str,
    system_prompt: str | None = None, user_prompt: str | None = None
) -> dict:
    """
    Executes the full inference pipeline, performing automatic speech recognition (ASR),
    text summarization, and named entity recognition (NER) based on the specified model
    selection. This function handles processing of an uploaded audio file by saving it
    temporarily, extracting text from the audio, summarizing the text, and identifying named
    entities in the text. It also cleans up temporary artifacts after execution.

    Args:
        file (UploadedFile): The uploaded file object to be processed. It represents an
            audio file that will undergo speech-to-text extraction.
        model_selection (str): Indicates the chosen pipeline for text summarization and
            named entity recognition. Acceptable values include "T5 + GliNER", "Bart +
            GliNER", or others for an alternative model.
        temp_dir (str): The directory path for temporary storage of the uploaded audio file
            during processing.
        system_prompt (str | None, optional): The system-level prompt to be applied for
            inference in the selected pipeline. Defaults to None.
        user_prompt (str | None, optional): The user-specific prompt to be applied for
            inference in the selected pipeline. Defaults to None.

    Returns:
        dict: A dictionary containing the processed results including the text summary and
            named entities along with corresponding scores where relevant.
    """

    # Create an instance of Whisper model
    whisper_model = whisper_ai.WhisperAI(
        model_name=cf.MODELS_CONFIG["Whisper_AI_Configurations"]["Model_Name"],
        model_task="automatic-speech-recognition",
        device=cf.DEVICE,
        token_required=cf.MODELS_CONFIG["Whisper_AI_Configurations"]["Hugging_Face_Token"],
        token=None
    )

    # Save the uploaded file into a temporary folder for Streamlit
    temp_folder = cf.DATA_PATH.joinpath(temp_dir).resolve()
    os.makedirs(temp_folder, exist_ok=True)

    temp_file = temp_folder.joinpath(file.name).resolve()
    with open(temp_file, "wb") as f:
        f.write(file.read())

    # Run text extraction through Whisper AI
    extracted_text = whisper_model.inference(
        audio_files=[temp_file.__str__()],
        max_new_tokens=cf.MODELS_CONFIG["Whisper_AI_Configurations"]["Maximum_Token_Generation"],
        language=cf.MODELS_CONFIG["Whisper_AI_Configurations"]["Language_Selection"]
    )[0]["text"]

    # Run text summarization inference pipeline based on model selection
    if model_selection == "T5 + GliNER":
        # Text summary
        model = google_flan.GoogleFlanT5(
            model_name = cf.MODELS_CONFIG["Google_Flan_T5"]["Model_Name"],
            device = cf.DEVICE,
            token_required = cf.MODELS_CONFIG["Google_Flan_T5"]["Hugging_Face_Token"],
            token = None
        )
        text_summary = model.inference(
            input_text = f"summarize: {extracted_text}",
            min_length = cf.MODELS_CONFIG["Google_Flan_T5"]["Maximum_Token_Generation"]["Medium_Output"]["Minimum_Length"],
            max_length = cf.MODELS_CONFIG["Google_Flan_T5"]["Maximum_Token_Generation"]["Medium_Output"]["Maximum_Length"]
        )

        # Clean the output
        text_summary = text_summary.replace("<pad>", "").replace("</s>", "").replace("summary:", "").strip()

    elif model_selection == "Bart + GliNER":
        # Text summary
        text_summary = bart_inference(text=extracted_text, model_ident="Facebook_Bart_CNN")

    else:
        model_output = phi4_inference(
            text=extracted_text,
            model_ident="Phi4_Language_Model",
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # Clean the output and extract the summary out
        cleaned_dict = json_utils.json_reformatting(model_output)
        text_summary = cleaned_dict["SUMMARY"]

    # Named entities extraction
    if model_selection == "T5 + GliNER" or model_selection == "Bart + GliNER":
        model = GLiNER.from_pretrained(
            pretrained_model_name_or_path=cf.MODELS_CONFIG["Gliner_Model"]["Model_Name"],
            load_tokenizer=True,
            max_length=cf.MODELS_CONFIG["Gliner_Model"]["Maximum_Length"]
        ).to("cuda" if cf.DEVICE == "auto" else cf.DEVICE).eval()
        labels = cf.MODELS_CONFIG["Gliner_Model"]["Labels"]
        entities = model.predict_entities(extracted_text, labels)

        # Flatten the dictionary and calculate the average score for each entity
        scored_entities = streamlit_utils.ner_cleaning(entities)

    else:
        scored_entities = {k: v for k, v in cleaned_dict.items() if k != "SUMMARY"}

    # Combined everything and return the dictionary
    combined_dict = {"SUMMARY": text_summary, **scored_entities}

    # Clean up the artifacts
    shutil.rmtree(temp_folder)
    return combined_dict
