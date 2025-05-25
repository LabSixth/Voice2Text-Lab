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
_model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")

def extract_entities(text: str, labels: list[str] | None = None) -> dict:
    """Extract named entities from the text using GLiNER.
    Args:
        text (str): The input text from which to extract entities.
        labels (list[str], optional): List of labels to filter the entities. Defaults to None, which uses all labels.
    Returns:
        dict: A dictionary containing the extracted entities categorized by their labels.
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
