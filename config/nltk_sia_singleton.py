from nltk.sentiment import SentimentIntensityAnalyzer


class NltkSiaSingleton:
    _instance = None
    sia = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.sia is None:
            self.sia = SentimentIntensityAnalyzer()
