import ollama


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