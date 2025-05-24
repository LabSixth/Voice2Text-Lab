
import time
from typing import Generator


def stream_text(text: str) -> Generator[str, None, None]:
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)


def calculate_ner_cof(text_list: list[str], text_score: list[float]) -> dict:
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
