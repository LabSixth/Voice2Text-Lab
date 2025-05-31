
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, BatchEncoding

class GoogleFlanT5:

    def __init__(self, model_name: str, device: str, token_required: bool = False, token: str | None = None):
        """
        Initializes the class to manage a model and its tokenizer, utilizing the
        specified model name, device, and optional token for authorization.

        Args:
            model_name: The name of the pretrained model to be loaded.
            device: The computational device to use, like "cpu" or "cuda".
            token_required: Flag indicating if a token is required for loading
                the model.
            token: The optional authorization token to authenticate access to
                the model resources.
        """

        self.model_name = model_name
        self.device = device
        self.token_required = token_required
        self.token = token
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device,
            trust_remote_code=True,
            token=token if token_required else None
        )
        self.model = T5ForConditionalGeneration.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
            token=token if token_required else None
        ).eval()

    def inference(self, input_text: str, min_length: int, max_length: int) -> str:
        """
        Generates text based on the provided input, with constraints on minimum and
        maximum length.

        Args:
            input_text (str): Text input that serves as the basis for generating
                output text.
            min_length (int): Minimum length of the generated output. Ensures that
                the generated text is at least this many tokens long.
            max_length (int): Maximum length of the generated output. Ensures that
                the generated text does not exceed this number of tokens.

        Returns:
            str: The generated text based on the input_text and specified length
            constraints.
        """

        # Generate embeddings from the input text given
        input_ids = self.tokenizer(
            input_text,
            return_tensors="pt"
        ).input_ids.to("cuda" if self.device == "auto" else self.device)

        # Run inference
        with torch.no_grad():
            outputs = self.model.generate(input_ids, min_length=min_length, max_length=max_length)

        # Decode the output and return
        decoded_outputs = self.tokenizer.decode(outputs[0])
        return decoded_outputs
