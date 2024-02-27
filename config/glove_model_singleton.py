from gensim.models import KeyedVectors


class GloveModelSingleton:
    _instance = None
    model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.model is None:
            word2vec_output_file = '../glove.twitter.27B/glove.twitter.27B.25d.word2vec.txt'
            self.model = KeyedVectors.load_word2vec_format(word2vec_output_file, binary=False)
