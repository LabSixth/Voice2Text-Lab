import whisper

_models = {}

def transcribe_base(audio) -> str:
    """Transcribe audio using the base Whisper model.
    Args:
        audio (str or UploadedFile): Path to the audio file or an UploadedFile object.
    Returns:
        str: The transcribed text.
    """

    return _transcribe(audio, "base")


def transcribe_small(audio) -> str:
    """Transcribe audio using the small Whisper model.
    Args:
        audio (str or UploadedFile): Path to the audio file or an UploadedFile object.
    Returns:
        str: The transcribed text.
    """

    return _transcribe(audio, "small")


def _transcribe(audio, model_name: str) -> str:
    """Transcribe audio using the specified Whisper model.
    Args:
        audio (str or UploadedFile): Path to the audio file or an UploadedFile object.
        model_name (str): The name of the Whisper model to use.
    Returns:
        str: The transcribed text.
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