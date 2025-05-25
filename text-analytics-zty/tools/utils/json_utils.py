
import re
import json


def json_reformatting(text: str) -> dict:
    """
    Extracts and parses a JSON block from a given text. If no such block exists,
    the method will attempt to parse the entire text as JSON. This function is
    useful for reformatting raw input that includes embedded JSON content.

    Args:
        text: A string input containing either a single JSON block enclosed in
            triple backticks (` ```json ... ``` `) or a raw JSON string.

    Returns:
        dict: A dictionary parsed from the extracted or entire JSON content.
    """

    # Look for a JSON block enclosed in triple backticks
    match = re.search(r'```json\n(.*?)\n```', text, re.DOTALL)

    # If there is a match of the pattern, extract the JSON block and parse it
    if match:
        json_str = match.group(1)
        data = json.loads(json_str)

    # Otherwise, load the original text into JSON object
    else:
        data = json.loads(text)

    return data
