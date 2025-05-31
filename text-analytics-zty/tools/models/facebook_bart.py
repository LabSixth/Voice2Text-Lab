
from transformers import pipelines


class FacebookBart:

    def __init__(self, model_name: str, device: str, task: str, token_required: bool = False, token: str | None = None):
        """
        Initializes a new instance of a class with specified model, device, task,
        and optional token configurations. It sets up a pipeline for performing the
        defined task using the specified model and device.

        Args:
            model_name: The name of the model to be loaded.
            device: The hardware device configuration for the pipeline, such as CPU
                or GPU.
            task: The specific task that the pipeline will handle, such as
                "text-classification" or "translation".
            token_required: Indicates whether an authentication token is required for
                accessing or initializing the model. Defaults to False.
            token: Optional authentication token used if token_required is set to
                True.
        """

        self.model_name = model_name
        self.device = device
        self.task = task
        self.token_required = token_required
        self.token = token
        self.pipe = pipelines.pipeline(
            task=task,
            model=model_name,
            device_map=device,
            torch_dtype="auto",
            trust_remote_code=True,
            token=token if token_required else None
        )

    def inference(self, input_text: str, min_length: int, max_length: int) -> str:
        """
        Runs a text summarization inference using the pre-defined pipeline. Generates a summarized
        version of the input text constrained by the specified minimum and maximum lengths.

        Args:
            input_text: A string representing the text to be summarized.
            min_length: An integer representing the minimum allowable length of the summary.
            max_length: An integer representing the maximum allowable length of the summary.

        Returns:
            A string containing the summarized version of the input text.
        """

        output_text = self.pipe(input_text, min_length=min_length, max_length=max_length)
        return output_text[0]["summary_text"]
