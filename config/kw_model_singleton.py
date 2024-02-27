from keybert import KeyBERT


class KwModelSingleton:
    _instance = None
    model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.model is None:
            self.model = KeyBERT()
