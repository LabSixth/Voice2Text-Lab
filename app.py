import os
import tempfile

import streamlit as st
import pandas as pd
import yaml

from utils import (
    enhance_vocals,
    extract_vocals,
    remove_silence,
    transcribe_vocals,
    summarize_text,
)

# load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

st.set_page_config(page_title="What are they saying?", page_icon="🗣️")
st.title("🗣️ What are they saying?")

uploaded = st.file_uploader("Upload WAV/MP3", type=["wav","mp3"])
model_sel = st.selectbox("Pick you favorite summarizer model", cfg["summary_models"])
enhance   = st.checkbox("Enhance vocals", value=True)
run       = st.button("Run")

if run and uploaded:
    with tempfile.TemporaryDirectory() as td:
        # 1) Save upload
        inp = os.path.join(td, uploaded.name)
        with open(inp, "wb") as f:
            f.write(uploaded.read())

        # 2) Enhance
        if enhance:
            st.info("🔊 Enhancing vocals…")
            base, _ = os.path.splitext(inp)
            enh = f"{base}_enhanced.wav"
            enhance_vocals(inp, enh)
        else:
            enh = inp

        # 3) Extract
        st.info("🎤 Extracting vocals…")
        vd = os.path.join(td, "vocals")
        os.makedirs(vd, exist_ok=True)
        extract_vocals(enh, vd)

        # 4) Remove silence
        st.info("🤫 Removing silence…")
        for root, _, files in os.walk(vd):
            for fn in files:
                if fn.endswith(".wav"):
                    p   = os.path.join(root, fn)
                    out = p.replace(".wav", "_clean.wav")
                    remove_silence(p, out)

        # 5) Transcribe
        st.info("📝 Transcribing…")
        csv_out = os.path.join(td, "transcripts.csv")
        transcribe_vocals(vd, csv_out)

        # 6) Summarize + NER only
        df = pd.read_csv(csv_out)
        st.info("🪄 Summarizing & NER…")
        df = summarize_text(df, model_sel)

        # 7) Display & download
        for _, row in df.iterrows():
            st.subheader(f"🎵 {row['song_name']}")
            st.text_area("Transcript", row["transcript"], height=120)
            st.markdown(f"**Entities:** {row['entities']}")
            st.markdown(f"**Long Summary:** {row['long_summary']}")
            st.markdown(f"**Short Summary:** {row['short_summary']}")
            st.markdown(f"**Tiny Summary:** {row['tiny_summary']}")

        st.download_button(
            "📥 Download CSV",
            data=df.to_csv(index=False),
            file_name="results.csv",
            mime="text/csv"
        )
        st.success("Hope you had fun!")
