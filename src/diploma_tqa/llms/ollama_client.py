import ollama


MISTRAL_MEDIUM_35 = "mistral-medium-3.5:128b-q4_K_M"


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
        # Preserve the original Qwen request exactly.
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
            think="False",
            stream=False,
            options={
                "num_ctx": 32768,
                "temperature": 0.0,
                "num_predict": 2048,
            },
        )
        return response["response"]