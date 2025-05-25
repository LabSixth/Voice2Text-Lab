
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


def ner_cleaning(extracted_ner: list[dict]) -> dict:
    """
    Cleans and processes a list of Named Entity Recognition (NER) dictionaries by grouping entities
    based on their labels and calculating the average confidence score for each label.

    This function organizes extracted NER information into corresponding categories based on
    entity labels, calculates the average score for each type of entity, and transforms the
    labels to uppercase for the final output.

    Args:
        extracted_ner (list[dict]): A list of dictionaries where each dictionary represents an NER
            entity with "label" (str), "text" (str), and "score" (float) as keys.

    Returns:
        dict: A dictionary where the keys are uppercased entity labels (str) and the values are
            the average confidence scores (float) for those labels. Returns an empty dictionary
            if the input list is empty.
    """

    # Loop through the list of dictionary and calculate average score for each type of NER
    if extracted_ner:
        # Separate the entities and scores into different bucket
        entities = {}
        scores = {}
        for entity in extracted_ner:
            if entity["label"] in entities:
                entities[entity["label"]].append(entity["text"])
                scores[entity["label"]].append(entity["score"])
            else:
                entities[entity["label"]] = [entity["text"]]
                scores[entity["label"]] = [entity["score"]]

        # For each of the entity, calculate the average score
        all_scores = {}
        for key, value in entities.items():
            average_score = calculate_ner_cof(value, scores[key])
            all_scores[key.upper()] = average_score

        return all_scores

    else:
        return {}

