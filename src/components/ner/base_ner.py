class BaseNER:
    def __init__(self):
        pass

    def run(self, text: str, type=None):
        raise NotImplementedError("Subclasses should implement this method.")
    
    def run_mult(self, texts: list[str], type=None):
        responses = []
        for text in texts:
            response = self.run(text, type)
            responses.append(response)
        return responses
    