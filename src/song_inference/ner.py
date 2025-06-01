
from gliner import GLiNER

# Pronouns to filter out
_PRONOUNS = {"i", "you", "we", "they", "it", "he", "she"}
# Map GLiNER labels to app categories
_LABEL_MAP = {
    "person": "CHARACTERS",
    "location": "LOCATIONS",
    "event": "OBJECTS",
    "product": "OBJECTS"
}
DEFAULT_LABELS = list(_LABEL_MAP.keys())

# Load GLiNER model once
_model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5")


def extract_entities(text: str, labels: list[str] | None = None) -> dict:
    """
    Extracts named entities from the given text using a predefined list of labels or default labels.

    This function utilizes a prediction model to extract entities categorized into various predefined
    categories. Each recognized entity is scored and added to the appropriate category, skipping
    pronouns and undefined labels.

    Args:
        text: The string input containing the text data for entity extraction.
        labels: A list of strings representing label categories to filter entities.
            If None, default labels will be used.

    Returns:
        A dictionary where keys are entity categories, and values are lists of dictionaries,
        each containing the extracted entity text and a confidence score.
    """

    if labels is None:
        labels = DEFAULT_LABELS

    # Prepare output
    entities = {category: [] for category in _LABEL_MAP.values()}
    # Predict GLiNER
    raw = _model.predict_entities(text, labels)

    for item in raw:
        label = item["label"]
        ent_text = item["text"].strip()
        if ent_text.lower() in _PRONOUNS:
            continue
        category = _LABEL_MAP.get(label)
        if category:
            entities[category].append({
                "text": ent_text,
                "score": item.get("score", 0)
            })
    return entities