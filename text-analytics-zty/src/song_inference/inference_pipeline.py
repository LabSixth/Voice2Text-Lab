from .separate import separate_vocals
from .transcribe import transcribe_base, transcribe_small
from .ner import extract_entities
from .summarize import summarize_bart, summarize_t5


def full_inference_pipeline(file,
                            transcription_model: str = "base",
                            summary_model: str = "bart",
                            extract_vocals: bool = False) -> dict:
    """Run the full inference pipeline.
    Args:
        file (str): Path to the audio file.
        transcription_model (str): Model to use for transcription. Options: "base", "small".
        summary_model (str): Model to use for summarization. Options: "bart", "t5".
        extract_vocals (bool): Whether to extract vocals from the audio file.
    Returns:
        dict: Dictionary containing the transcript, long summary, short summary, tiny summary, and named entities.
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
    long_sum  = summarize_fn(transcript, mode="long")
    short_sum = summarize_fn(transcript, mode="short")
    tiny_sum  = summarize_fn(transcript, mode="tiny")

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