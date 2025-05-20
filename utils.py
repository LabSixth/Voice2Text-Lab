import os
import subprocess
import wave
import yaml
import re
from typing import List

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import webrtcvad
import whisper
from spleeter.separator import Separator
from transformers import pipeline

# load config
with open("config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

PRE_EMPHASIS           = cfg["pre_emphasis"]
SILENCE_AGGRESSIVENESS = cfg["silence_aggressiveness"]
PCM_SAMPLE_RATE        = cfg["pcm_sample_rate"]
MODEL_SIZE             = cfg["model_size"]


def enhance_vocals(input_file: str, output_file: str):
    """Apply a simple pre-emphasis filter to boost vocals."""
    y, sr = librosa.load(input_file, sr=None)
    emphasized = np.append(y[0], y[1:] - PRE_EMPHASIS * y[:-1])
    sf.write(output_file, emphasized, sr)


def extract_vocals(input_file: str, output_dir: str):
    """Run Spleeter 2-stem separation (vocals + accompaniment)."""
    sep = Separator("spleeter:2stems")
    sep.separate_to_file(input_file, output_dir)


def remove_silence(input_file: str, output_file: str):
    """Use WebRTC VAD to strip non-speech and write a clean WAV."""
    vad = webrtcvad.Vad(SILENCE_AGGRESSIVENESS)

    # convert to 16 kHz mono PCM
    pcm = input_file.replace(".wav", "_pcm.wav")
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-ac", "1", "-ar", str(PCM_SAMPLE_RATE),
        "-sample_fmt", "s16", pcm
    ]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    frames = []
    with wave.open(pcm, "rb") as wf:
        frame_bytes = int(PCM_SAMPLE_RATE * 0.03) * wf.getsampwidth()
        while True:
            buf = wf.readframes(frame_bytes // wf.getsampwidth())
            if not buf:
                break
            if len(buf) == frame_bytes and vad.is_speech(buf, PCM_SAMPLE_RATE):
                frames.append(buf)

    with wave.open(output_file, "wb") as wf_out:
        wf_out.setparams((1, 2, PCM_SAMPLE_RATE, 0, "NONE", "not compressed"))
        wf_out.writeframes(b"".join(frames))

    os.remove(pcm)


def transcribe_vocals(vocals_dir: str, out_csv: str):
    """
    Walk vocals_dir for *vocals_clean.wav* files, run Whisper,
    and save one row per track into out_csv.
    """
    model = whisper.load_model(MODEL_SIZE)
    rows = []
    for root, _, files in os.walk(vocals_dir):
        for fn in files:
            if not fn.endswith("vocals_clean.wav"):
                continue
            path = os.path.join(root, fn)
            song = os.path.basename(root)
            res = model.transcribe(
                path,
                language="en",
                temperature=0.2,
                beam_size=5,
            )
            rows.append({
                "song_name": song,
                "transcript": res["text"].strip()
            })
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def split_into_sentences(text: str) -> List[str]:
    """Naïve sentence split on [.?!] boundaries."""
    return [s.strip() for s in re.split(r'(?<=[.?!])\s+', text) if s.strip()]


def truecase_and_punctuate(text: str) -> str:
    """Tiny pseudo-punctuator: split on '.', capitalize, re-join."""
    parts = [p.strip() for p in text.split('.') if p.strip()]
    if not parts:
        return text
    sent = '. '.join(p.capitalize() for p in parts)
    return sent + '.'


def summarize_text(df: pd.DataFrame, model_name: str="t5-small") -> pd.DataFrame:
    """
    For each transcript:
      1) Pseudo-punctuate + split into sentences
      2) Run NER with dslim/bert-base-NER
      3) If no entities, leave as "None"
      4) Generate long/short/tiny summaries
    """
    device = 0 if torch.cuda.is_available() else -1
    summarizer = pipeline("summarization", model=model_name, device=device)
    ner_pipe = pipeline(
        "token-classification",
        model=cfg["ner_model"],
        tokenizer=cfg["ner_model"],
        aggregation_strategy=cfg["ner_agg"],
        device=device,
    )

    # prepare output columns
    for col in ("entities", "long_summary", "short_summary", "tiny_summary"):
        df[col] = ""

    for i, row in df.iterrows():
        canon = truecase_and_punctuate(row["transcript"])

        # run NER on each sentence
        raw_ents = []
        for sent in split_into_sentences(canon):
            raw_ents.extend(ner_pipe(sent))

        # filter by label & confidence
        filtered = [
            e for e in raw_ents
            if e["score"] >= cfg["ner_thresh"]
               and e["entity_group"] in {"PER","ORG","LOC","MISC"}
        ]
        keep = {e["word"] for e in filtered}

        # set entities field
        df.at[i, "entities"] = ", ".join(sorted(keep)) or "None"

        # generate summaries
        df.at[i, "long_summary"]  = summarizer(
            canon, max_length=250, min_length=100, do_sample=False
        )[0]["summary_text"].strip()

        df.at[i, "short_summary"] = summarizer(
            canon, max_length=100, min_length=30, do_sample=False
        )[0]["summary_text"].strip()

        df.at[i, "tiny_summary"]  = summarizer(
            canon, max_length=50, min_length=10, do_sample=False
        )[0]["summary_text"].strip()

    return df
