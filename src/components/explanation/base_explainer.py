import igraph as ig

class BaseExplainer:
    def __init__(self):
        pass

    def run(self, question: str, answer: str, qna_context_prefix = "", qna_context="", context_prefix="", context="") -> str:
        raise NotImplementedError("Subclasses should implement this method.")