
import streamlit as st
import time
from src.speech_inference import pre_compute
from tools.utils import streamlit_utils


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