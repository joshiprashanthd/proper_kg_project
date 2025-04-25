from src.utils import OpenAIModel, BaseModel, logger
class BaseNER:
    def __init__(self, model: BaseModel = None):
        self.model = OpenAIModel("gpt-4o-mini", 'cuda') if not model else model
        logger.info(f"Using OpenAI model: {self.model.model_name}")

    def run(self, text: str, type=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def run_mult(self, texts: list[str], type=None):
        responses = []
        for text in texts:
            response = self.run(text, type)
            responses.append(response)
        return responses
    