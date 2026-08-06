import ollama


MISTRAL_MEDIUM_35 = "mistral-medium-3.5:128b-q4_K_M"
QWEN_36_27B_BF16 = "qwen3.6:27b-bf16"


class OllamaClient:
    def __init__(
        self,
        model: str = "qwen2.5-coder:1.5b",
        temperature: float = 0.0,
        num_predict: int = 256,
    ):
        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict

    def generate(self, prompt: str) -> str:
        # Qwen3.6: enable thinking, preserve temperature 0.
        if self.model == QWEN_36_27B_BF16:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                think=True,
                stream=False,
                options={
                    "temperature": 0.0,
                    "num_predict": 8192,
                },
            )
            return response["response"]

        # Preserve the original Qwen2.5 request exactly.
        if self.model != MISTRAL_MEDIUM_35:
            response = ollama.generate(
                model=self.model,
                prompt=prompt,
                stream=False,
                options={
                    "temperature": self.temperature,
                    "num_predict": self.num_predict,
                },
            )
            return response["response"]

        # Mistral-specific configuration.
        response = ollama.generate(
            model=self.model,
            prompt=prompt,
            think=False,
            stream=False,
            options={
                "num_ctx": 32768,
                "temperature": 0.0,
                "num_predict": 2048,
            },
        )
        return response["response"]