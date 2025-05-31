
import streamlit as st
from src import global_configs as cf
from src.speech_inference import pre_compute, text_inference
from tools.utils import streamlit_utils, json_utils
from src.song_inference.inference_pipeline import full_inference_pipeline


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
                    if summary["PERSONS"]:
                        st.write(summary["PERSONS"])
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


# --------------------------------------------------------------------------------------------------------------------
# User Uploaded Custom Speech - Summarization + NER
# --------------------------------------------------------------------------------------------------------------------

with st.expander(label="Speech to Summary + NER - Your Own Speech File 🔊"):
    # Write a short description
    st.markdown(
        """
        In this section of the application, you can choose to upload your own audio file and get summarized text and 
        named entities extracted from it. To get started, upload an audio file, choose a model, and click on 
        the "Summarize" button.
        """
    )

    uploaded_file = st.file_uploader(label="Upload your audio file here", type=["wav", "mp3", "flac"])
    model_option = st.selectbox(
        label="Model selection",
        options=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["User_Speech_Upload_Options"],
        placeholder="Select a model to try...",
        index=None
    )
    run = st.button(label="Run Pipeline")

    # Once a file has been uploaded and a model has been selected, process the file and extract summary and NER
    if uploaded_file and model_option and run:
        # Check if prompts are needed
        system_prompt = (
            cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Summarization_Prompts"]["System_Prompt"]
            if model_option == "Phi4 Language Model"
            else None
        )
        user_prompt = (
            cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Summarization_Prompts"]["User_Prompt"]
            if model_option == "Phi4 Language Model"
            else None
        )

        # Run inference based on the option selected by user
        output = text_inference.full_inference_pipeline(
            file=uploaded_file,
            model_selection=model_option,
            temp_dir=cf.STREAMLIT_CONFIG["Streamlit_Application_Configurations"]["Streamlit_Temp_Folder"],
            system_prompt=system_prompt,
            user_prompt=user_prompt
        )

        # From the output, display the summary and NER
        st.markdown("\n\nA summary of the audio file provided is as follows.")
        summary_container = st.container(border=True)
        with summary_container:
            st.write_stream(streamlit_utils.stream_text(output["SUMMARY"]))

        st.markdown("\n\nFrom the audio file provided, the extracted entities are as follows.")
        persons, locations, organizations = st.columns(3, border=True)
        with persons:
            st.markdown("**The Persons**")
            try:
                if output["PERSONS"]:
                    st.write(output["PERSONS"])
                else:
                    st.write("No person is detected from the audio file provided.")
            except KeyError:
                st.write("No person is detected from the audio file provided.")

        with locations:
            st.markdown("**The Locations**")
            try:
                if output["LOCATION"]:
                    st.write(output["LOCATION"])
                else:
                    st.write("No location is detected from the audio file provided.")
            except KeyError:
                st.write("No location is detected from the audio file provided.")

        with organizations:
            st.markdown("**The Organizations**")
            try:
                if output["ORGANIZATION"]:
                    st.write(output["ORGANIZATION"])
                else:
                    st.write("No organization is detected from the audio file provided.")
            except KeyError:
                st.write("No organization is detected from the audio file provided.")


# --------------------------------------------------------------------------------------------------------------------
# User Uploaded Custom Song - Summarization + NER
# --------------------------------------------------------------------------------------------------------------------
with st.expander(label="🎵 Your Song Inference"):
    st.markdown(
        """
        Upload your own song/audio and run the full VoiceMuse pipeline: transcription, summaries, and entities.
        """
    )
    uploaded_file = st.file_uploader(label="Upload your song file", type=["wav", "mp3", "flac"])
    extract_vocals = st.checkbox("Extract vocals (separate voice)")
    transcription_model = st.selectbox("Transcription Model", ["base", "small"], index=0)
    summary_model = st.selectbox(
        "Summary Model",
        ["bart", "t5"],
        format_func=lambda x: "BART (facebook/bart-large-cnn)" if x=="bart" else "T5 (t5-small)"
    )
    run = st.button("Run Song Pipeline")

    if uploaded_file and run:
        output = full_inference_pipeline(
            file=uploaded_file,
            transcription_model=transcription_model,
            summary_model=summary_model,
            extract_vocals=extract_vocals
        )
        # Transcript
        st.subheader("Transcript")
        st.text_area("", value=output["TRANSCRIPT"], height=200)
        # Entities
        st.subheader("Named Entities")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Characters**")
            for e in output.get("CHARACTERS", []): st.write(f"{e['text']} ({e['score']:.2f})")
        with c2:
            st.markdown("**Locations**")
            for e in output.get("LOCATIONS", []): st.write(f"{e['text']} ({e['score']:.2f})")
        with c3:
            st.markdown("**Objects**")
            for e in output.get("OBJECTS", []): st.write(f"{e['text']} ({e['score']:.2f})")
        # Summaries
        st.subheader("Summaries")
        cols = st.columns(3)
        for col, title, key in zip(cols, ["Long","Short","Tiny"], ["LONG_SUMMARY","SHORT_SUMMARY","TINY_SUMMARY"]):
            with col:
                st.markdown(f"**{title} Summary**")
                st.write(output.get(key, ""))