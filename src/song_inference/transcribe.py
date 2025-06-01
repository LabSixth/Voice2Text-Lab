
import whisper

_models = {}


def transcribe_base(audio) -> str:
    """
    Transcribes audio input using the "base" transcription model.

    This function utilizes a lower-level transcription mechanism to process
    the given audio input. The "base" model is a default setting that provides
    a balanced trade-off between speed and accuracy, making it suitable for a
    variety of transcription tasks. The function returns the transcription result
    as a string.

    Args:
        audio: Input audio data to be transcribed.

    Returns:
        str: Transcription of the provided audio input.
    """

    return _transcribe(audio, "base")


def transcribe_small(audio) -> str:
    """
    Transcribes audio data into text using a smaller model for efficient processing.

    This function utilizes a predefined small model to convert the
    provided audio data into a text transcription. It is designed
    for scenarios where resource efficiency is critical while maintaining
    an acceptable level of accuracy.

    Args:
        audio: The audio data input for transcription.

    Returns:
        str: The transcribed text derived from the input audio.
    """

    return _transcribe(audio, "small")


def _transcribe(audio, model_name: str) -> str:
    """
    Transcribes the given audio input using a specified Whisper model. If the model
    is not already loaded, it is initialized. The function accepts both file paths
    and file-like objects as audio input.

    Args:
        audio: Input audio, either as a file path (str) or as a file-like object
            with a readable `read` method and a `name` attribute.
        model_name: The name of the Whisper model to be used for transcription.

    Returns:
        str: The transcribed text from the audio input.
    """

    if model_name not in _models:
        _models[model_name] = whisper.load_model(model_name)
    model = _models[model_name]
    if hasattr(audio, 'read'):
        tmp = f"/tmp/{audio.name}"
        with open(tmp, 'wb') as f:
            f.write(audio.read())
        audio_path = tmp
    else:
        audio_path = audio
    return model.transcribe(audio_path)["text"]