
import time
from typing import Generator


def stream_text(text: str) -> Generator[str, None, None]:
    """
    Yields words from a given text string one at a time, adding a trailing space to each word,
    and includes a delay between yielding each word.

    Args:
        text: The input string that will be split into words and streamed.

    Yields:
        str: A word from the text with a trailing space.
    """

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def calculate_ner_cof(text_list: list[str], text_score: list[float]) -> dict:
    """
    Calculates the average score for each unique text item in the provided list by
    aggregating their scores from the corresponding list of scores.

    This function uses two dictionaries, one for storing the sum of scores and another
    for maintaining the count of occurrences for each text item, to compute the average
    score of each unique text entry. The averaging process is case-insensitive.

    Args:
        text_list (list[str]): A list of text items for which the average score needs
            to be calculated. The text items are processed case-insensitively.
        text_score (list[float]): A list of numerical scores corresponding to each
            text item in the text_list.

    Returns:
        dict: A dictionary where keys are unique text items (in lowercase) from the
        `text_list` and values are the respective average scores calculated based
        on entries in `text_score`.
    """

    # Create two dictionaries, one for counts and one for score
    scores = {}
    counts = {}

    # Loop through the dictionary and gather the statistics to calculate average score
    for idx, value in enumerate(text_list):
        value = value.lower()

        if value in scores:
            scores[value] += text_score[idx]
            counts[value] += 1
        else:
            scores[value] = text_score[idx]
            counts[value] = 1

    # Calculate the average score and return
    average_scores = {key: scores[key] / counts[key] for key in scores}
    return average_scores
