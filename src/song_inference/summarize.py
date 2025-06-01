
from transformers import pipeline

_summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
_summarizer_t5 = pipeline("summarization", model="t5-small", device=-1)
LENGTHS = {"long": (200, 512), "short": (75, 150), "tiny": (15, 30)}


def summarize_bart(text: str, mode: str = "short") -> str:
    """
    Summarizes a given text using a BART model, with the summary length specified by
    the provided mode. The function ensures that the processed snippet length does
    not exceed a predefined character limit, and it utilizes a pre-configured
    summarization function for generating the summary.

    Args:
        text: The input text to be summarized.
        mode: The desired summary length. Should be one of 'long', 'short', or 'tiny'.

    Returns:
        A string containing the generated summary of the input text.

    Raises:
        ValueError: If the provided mode is not one of 'long', 'short', or 'tiny'.
    """

    if mode not in LENGTHS:
        raise ValueError("Mode must be 'long','short','tiny'.")
    min_len, max_len = LENGTHS[mode]
    snippet = text[:10240]
    return _summarizer_bart(snippet, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]


def summarize_t5(text: str, mode: str = "short") -> str:
    """
    Summarizes the input text using a T5 model based on the specified length mode.

    The function takes an input text, determines the summarization limits (minimum
    and maximum lengths) based on the mode provided, and processes the text for
    summarization. If the input text exceeds the model's token limit, it is trimmed
    to fit. The T5 model generates a summary of the specified size by using
    predefined constraints.

    Args:
        text: The text that needs to be summarized. It can be any long passage
            or content that fits the model's input limit.
        mode: Optional; The summarization mode determining the length of the
            summary. Allowed values are 'short', 'long', or 'tiny'. Defaults
            to 'short'.

    Raises:
        ValueError: If the provided mode is not one of the allowed values
            ('long', 'short', 'tiny').

    Returns:
        The summarized text as a string that is constrained by the length
        defined in the provided mode.
    """

    if mode not in LENGTHS:
        raise ValueError("Mode must be 'long','short','tiny'.")
    min_len, max_len = LENGTHS[mode]
    snippet = text[:10240]
    return _summarizer_t5(snippet, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]