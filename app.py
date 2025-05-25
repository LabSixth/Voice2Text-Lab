
import streamlit as st
import time
from src import global_configs as cf
from src.speech_inference import pre_compute, text_inference
from tools.utils import streamlit_utils, json_utils


# --------------------------------------------------------------------------------------------------------------------
# Application Main Page
# --------------------------------------------------------------------------------------------------------------------

st.set_page_config(page_title="What are they saying?", page_icon="🗣", layout="wide")
st.title("🗣 What are they saying?")
st.markdown(
    """
    Welcome to VoiceMuse — where your voice finds its rhythm. This app transforms spoken words into written expression, 
    and from there into lyrical form. Whether you’re capturing a passing thought, shaping poetic lines, or simply 
    experimenting with creativity, VoiceMuse helps you turn raw speech into something structured, meaningful, and ready 
    to be heard. Speak freely, shape your story, and experience the flow from voice to verse.
    
    ---
    To get started, expand any one of the sections below and get started with the application. 🚀
    """
)

# --------------------------------------------------------------------------------------------------------------------
# Original Speech Application - Summarization and NER
# --------------------------------------------------------------------------------------------------------------------

with st.expander(label="Speech to Summary + NER - Default"):
    # Write a short description
    st.markdown(
        """
        In this section of the application, we have prepared a set of predefined and precomputed datasets and metrics
        that have been supplied by [Libri Speech Corpus](https://www.openslr.org/12). To get started, pick a number
        between 1 and 97.
        """
    )
    row_id = st.text_input(label="Pick a number between 1 to 97", max_chars=2)

    # Check and make sure that the user is providing the correct number
    if row_id:
        try:
            if int(row_id) < 1 or int(row_id) > 97:
                st.write(":red[Please pick a number between 1 to 97.]")

        except Exception:
            st.write(":red[Please provide a valid number between 1 to 97.]")

    # If the user has provided the correct number, go through the process to present the results
    if row_id and 1 <= int(row_id) <= 97:
        # Get pre-computed data before presenting
        data = pre_compute.main_extraction(int(row_id))

        # Present the original text and T5 summary
        st.text_area(
            label="The original transcripts from the number that you have select is as follow.",
            value=data["recording_transcriptions"][0],
            height=350
        )

        st.markdown("\nUsing T5 Model from Hugging Face, the summaries are as follow.")
        time.sleep(2)

        t5_short, t5_medium, t5_long = st.columns(3, border=True)
        with t5_short:
            st.markdown("**T5 Short Summary**")
            st.write_stream(streamlit_utils.stream_text(data["t5_short"][0]))

        with t5_medium:
            st.markdown("**T5 Medium Summary**")
            st.write_stream(streamlit_utils.stream_text(data["t5_medium"][0]))

        with t5_long:
            st.markdown("**T5 Long Summary**")
            st.write_stream(streamlit_utils.stream_text(data["t5_large"][0]))

        # Present NER - Persons, Locations, and Organizations
        st.markdown(
            """
            \nFrom the original text, we've used GliNER to extract Named Entities and their associated scores. Below is a
            list of Named Entities extracted from the original text
            """
        )

        persons, locations, organizations = st.columns(3, border=True)
        with persons:
            st.markdown("**The Persons**")
            if data["persons_text"][0]:
                st.write(streamlit_utils.calculate_ner_cof(data["persons_text"][0], data["persons_score"][0]))
            else:
                st.write("No person is detected using GliNER package.")

        with locations:
            st.markdown("**The Locations**")
            if data["location_text"][0]:
                st.write(streamlit_utils.calculate_ner_cof(data["location_text"][0], data["location_score"][0]))
            else:
                st.write("No location is detected using GliNER package.")

        with organizations:
            st.markdown("**The Organizations**")
            if data["org_text"][0]:
                st.write(streamlit_utils.calculate_ner_cof(data["org_text"][0], data["org_score"][0]))
            else:
                st.write("No organization is detected using GliNER package.")

        # Let the user choose alternative models to use for summary
        st.markdown(
            """
            ---
            Now that you have seen the summary from T5. Let's explore summaries produced by other more sophisticated models.
            To get started, from the dropdown box below, choose the model that you'd liked to try.
            """
        )
        model_option = st.selectbox(
            label="Language models selection",
            options=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Additional_Models"],
            placeholder="Select a model to try...",
            index=None
        )

        if model_option:
            if model_option == "Facebook_Bart_CNN":
                st.markdown("\n\nUsing Facebook's BART model, the summary of the original text is as follow.")
                summary = text_inference.bart_inference(data["recording_transcriptions"][0], model_option)

                bart_container = st.container(border=True)
                with bart_container:
                    st.write_stream(streamlit_utils.stream_text(summary))

            else:
                st.markdown("\n\nUsing Microsoft's Phi4 Mini model, the summary of the original text is as follow.")
                summary = text_inference.phi4_inference(
                    text=data["recording_transcriptions"][0],
                    model_ident=model_option,
                    system_prompt=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Summarization_Prompts"]["System_Prompt"],
                    user_prompt=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Summarization_Prompts"]["User_Prompt"]
                )

                # Clean the text and extract the JSON part out
                summary = json_utils.json_reformatting(summary)

                # Serve the results
                phi4_container = st.container(border=True)
                with phi4_container:
                    st.write_stream(streamlit_utils.stream_text(summary["SUMMARY"]))

                st.markdown(
                    """
                    \nUsing Phi4 Mini model, we could also attempt to extract the entities from the original text. The following
                    are the entities extracted from the original text using Phi4 Mini model.
                    """
                )

                persons, locations, organizations = st.columns(3, border=True)
                with persons:
                    st.markdown("**The Persons**")
                    if summary["PERSON"]:
                        st.write(summary["PERSON"])
                    else:
                        st.write("No person is detected using Phi4 Mini.")

                with locations:
                    st.markdown("**The Locations**")
                    if summary["LOCATION"]:
                        st.write(summary["LOCATION"])
                    else:
                        st.write("No location is detected using Phi4 Mini.")

                with organizations:
                    st.markdown("**The Organizations**")
                    if summary["ORGANIZATION"]:
                        st.write(summary["ORGANIZATION"])
                    else:
                        st.write("No organization is detected using Phi4 Mini.")

