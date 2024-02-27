from nltk.corpus import cmudict


class CmuSingleton:
    _instance = None
    cmu = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.cmu is None:
            self.cmu = cmudict.dict()
