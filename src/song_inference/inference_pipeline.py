
from src.song_inference.separate import separate_vocals
from src.song_inference.transcribe import transcribe_base, transcribe_small
from src.song_inference.ner import extract_entities
from src.song_inference.summarize import summarize_bart, summarize_t5


def full_inference_pipeline(
    file,
    transcription_model: str = "base",
    summary_model: str = "bart",
    extract_vocals: bool = False
) -> dict:
    """
    Executes a full pipeline involving optional vocal separation, transcription, text summarization,
    and named entity recognition (NER). The pipeline processes an input file and returns a consolidated
    output containing the transcription, multiple levels of summaries, and extracted entities.

    Args:
        file: The path or handle to the input audio file that will be processed in this pipeline.
        transcription_model: The transcription model to use. Defaults to "base".
        summary_model: The summarization model to utilize. Defaults to "bart".
        extract_vocals: A boolean flag indicating whether to apply vocals separation. Defaults to False.

    Returns:
        dict: A dictionary encompassing the transcription, summaries at different granularity levels,
              and named entities extracted from the transcription.
    """

    # 1. Optional vocals extraction
    if extract_vocals:
        file = separate_vocals(file)

    # 2. Transcription
    if transcription_model == "small":
        transcript = transcribe_small(file)
    else:
        transcript = transcribe_base(file)

    # 3. Summarization
    summarize_fn = summarize_bart if summary_model == "bart" else summarize_t5
    long_sum = summarize_fn(transcript, mode="long")
    short_sum = summarize_fn(transcript, mode="short")
    tiny_sum = summarize_fn(transcript, mode="tiny")

    # 4. Named Entity Recognition
    entities = extract_entities(transcript)

    # 5. Consolidate outputs
    return {
        "TRANSCRIPT": transcript,
        "LONG_SUMMARY": long_sum,
        "SHORT_SUMMARY": short_sum,
        "TINY_SUMMARY": tiny_sum,
        # merge NER result keys
        **entities
    }