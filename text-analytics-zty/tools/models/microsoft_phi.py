
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from src import global_configs as cf


class Phi4Instruct:

    def __init__(self, model_name: str, model_task: str, token_required: bool = False, token: str | None = None):
        """
        Initializes a language model with the specified parameters and sets up a tokenizer
        and pipeline for the given task.

        Args:
            model_name: Name of the pre-trained model to use for initialization.
            model_task: Task to be performed using the model (e.g., text generation).
            token_required: Indicates whether authentication token is required for
                using the model.
            token: Optional authentication token used in model initialization if
                token_required is True.

        Attributes:
            model_name: Stores the name of the pre-trained model being used.
            model_task: Stores the specified task for the model.
            token_required: Stores whether the authentication token is required for
                model usage.
            token: Stores the authentication token if required.
            tokenizer: Instance of the tokenizer initialized for the specified model.
            model: Loaded model for performing the specified task.
            pipe: Pipeline configured with the initialized model and tokenizer to
                handle the specified task.
        """

        self.model_name = model_name
        self.model_task = model_task
        self.token_required = token_required
        self.token = token
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name,
            trust_remote_code=True,
            token=token if token_required else None
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name,
            device_map=cf.DEVICE,
            torch_dtype="auto",
            trust_remote_code=True,
            token=token if token_required else None
        )
        self.pipe = pipeline(task=model_task, model=self.model, tokenizer=self.tokenizer)

    def inference(
        self, system_prompt: str | None, user_prompt: str,
        max_new_tokens: int, temperature: float, top_p: float
    ) -> str:
        """
        Performs inference using the provided prompts and generation arguments to produce
        structured text output. Leverages a message-based template and configurable
        generation parameters such as `temperature` and `top_p`.

        Args:
            system_prompt (str | None): Initial system message content. If None, a default
                system message is used which describes the function of the assistant.
            user_prompt (str): User-provided input message.
            max_new_tokens (int): Maximum number of tokens to generate in the output.
            temperature (float): Sampling temperature for generation. Higher values result
                in more diverse outputs.
            top_p (float): Nucleus sampling parameter which ensures only tokens with the
                top cumulative probability mass (up to `top_p`) are used for generation.

        Returns:
            str: The generated text output from the model based on the given prompts and
                generation parameters.
        """

        # Create the message template for Phi4
        if system_prompt is None:
            system_prompt = "You are a helpful assistant for turning request into structured JSON output."

        message = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Create arguments for the pipeline and run inference
        generation_args = {
            "max_new_tokens": max_new_tokens,
            "return_full_text": False,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True if temperature > 0 else None
        }
        output = self.pipe(message, **generation_args)

        return output[0]['generated_text']
