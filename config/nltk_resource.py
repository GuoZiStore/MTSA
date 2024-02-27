# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 20:43
# @Author  : HuangHonghao
# @File    : nltk_resource.py
# @Description :


def download_nltk_resource():
    import nltk
    nltk.download('wordnet')
    nltk.download('wordnet_ic')
    nltk.download('punkt')
    nltk.download('sentiwordnet')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('stopwords')
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    nltk.download('omw-1.4')
    nltk.download('opinion_lexicon')


download_nltk_resource()
