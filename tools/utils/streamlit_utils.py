
import time


def stream_text(text: str):
    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)
