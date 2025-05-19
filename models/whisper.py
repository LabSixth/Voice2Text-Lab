
from pathlib import Path
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from src import global_configs as cf


class WhisperAI:

    def __init__(
        self, model_name: str, model_task: str,
        token_required: bool = False, token: str | None = None
    ):
        """
        A class for initializing and configuring a model pipeline for speech sequence-to-sequence tasks.

        Attributes:
            model_name (str): The name or path of the pretrained model to be loaded.
            token_required (bool): Indicates whether a token is required for authentication.
            token (str | None): The authentication token to access the pretrained model, if required.
            model: Instance of AutoModelForSpeechSeq2Seq initialized with the pretrained model.
            processor: Instance of AutoProcessor initialized with the pretrained model.
            pipe: The inference pipeline configured for the specified task using the model and processor.
        """

        self.model_name = model_name
        self.token_required = token_required
        self.token = token
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=cf.DEVICE,
            torch_dtype="auto",
            trust_remote_code=True,
            token=token if token_required else None
        )
        self.processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            token=token if token_required else None
        )
        self.pipe = pipeline(
            task=model_task,
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            device_map=cf.DEVICE,
        )

    def inference(self, audio_files: list[str], max_new_tokens: int, language: str) -> list[str]:
        """
        Performs inference on a list of audio files using a pre-configured pipeline, generating text
        outputs for each audio file provided. This method processes the audio files in batches and
        adjusts the generation parameters such as language and maximum new tokens.

        Args:
            audio_files (list[str]): List of paths to the audio files to process.
            max_new_tokens (int): The maximum number of new tokens to generate during inference.
            language (str): The language for the inference process.

        Returns:
            list[str]: A list of generated text outputs corresponding to each audio file.
        """

        # Get the length of the list of audio files
        batch_size = len(audio_files)

        # Run inference and get results
        result = self.pipe(
            audio_files,
            generate_kwargs={"language": language, "max_new_tokens": max_new_tokens},
            batch_size=batch_size
        )
        return result
