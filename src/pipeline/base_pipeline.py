from abc import ABC, abstractmethod

class BasePipeline(ABC):
    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def add_component(self, component):
        pass

    @abstractmethod
    def remove_component(self, component):
        pass

    @abstractmethod
    def get_components(self):
        pass

    @abstractmethod
    def reset(self):
        pass