from transformers import pipeline

_summarizer_bart = pipeline("summarization", model="facebook/bart-large-cnn", device=-1)
_summarizer_t5  = pipeline("summarization", model="t5-small", device=-1)
LENGTHS = {"long": (200, 512), "short": (75, 150), "tiny": (15, 30)}

def summarize_bart(text: str, mode: str="short") -> str:
    """Summarize text using BART model.
    Args:
        text (str): The input text to summarize.
        mode (str): The length of the summary. Options: "long", "short", "tiny".
    Returns:
        str: The summarized text.
    """

    if mode not in LENGTHS:
        raise ValueError("Mode must be 'long','short','tiny'.")
    min_len, max_len = LENGTHS[mode]
    snippet = text[:10240]
    return _summarizer_bart(snippet, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]


def summarize_t5(text: str, mode: str="short") -> str:
    """Summarize text using T5 model.
    Args:
        text (str): The input text to summarize.
        mode (str): The length of the summary. Options: "long", "short", "tiny".
    Returns:
        str: The summarized text.
    """
    
    if mode not in LENGTHS:
        raise ValueError("Mode must be 'long','short','tiny'.")
    min_len, max_len = LENGTHS[mode]
    snippet = text[:10240]
    return _summarizer_t5(snippet, max_length=max_len, min_length=min_len, do_sample=False)[0]["summary_text"]