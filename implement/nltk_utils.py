# -*- coding: utf-8 -*-
# @Time    : 2023/10/4 20:12
# @Author  : HuangHonghao
# @File    : utils.py
# @Description :
import math
import os
import string

import nltk
import pandas as pd
from keybert import KeyBERT
from nltk import word_tokenize
from nltk.corpus import wordnet, opinion_lexicon
from nltk.sentiment import SentimentIntensityAnalyzer

from config import kw_model, sia

# 下载词典和模型数据（执行一次即可）
# nltk.download('vader_lexicon')
# nltk.download('wordnet')

# # 加载VADER情感分析器
# sia = SentimentIntensityAnalyzer()
# 加载情感词典
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())


def get_sentiment_words(sentence):
    # 分词
    words = word_tokenize(sentence)

    # 在情感词典中查找单词
    positive_matched_words = [word for word in words if word in positive_words]
    negative_matched_words = [word for word in words if word in negative_words]

    return positive_matched_words + negative_matched_words


def extract_emotion_words(text):
    emotion_words = []
    for word in word_tokenize(text):
        scores = sia.polarity_scores(word)
        if scores['compound'] > 0 or scores['compound'] < 0:
            print('{}: {}'.format(word, scores['compound']))
            emotion_words.append(word)
    return emotion_words


def get_synonyms(word):
    synonyms = []
    antonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            if lemma.name() != word:
                synonyms.append(lemma.name())
            if lemma.antonyms():
                antonyms.append(lemma.antonyms()[0].name())
    return list(set(synonyms)), list(set(antonyms))


def count_words(sentence):
    # 去除标点符号
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    # 使用 split() 方法将句子分割成单词
    words = sentence.split()
    # 返回单词的个数
    return words


def get_disturbance_list(sentence, threshold):
    words = count_words(sentence)
    keywords = kw_model.extract_keywords(sentence, top_n=len(words))
    n = math.ceil(len(keywords) * threshold)
    filtered_words = [word for word, score in keywords[:n]]
    return filtered_words


def get_disturbance_list_form_file(sentence, threshold=0.2):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_path, "asset", "sst2", "keyword.csv")
    data = pd.read_csv(str(file_path), header=None)


def nltk_predict(text):
    scores = sia.polarity_scores(text)
    compound_score = scores['compound']
    if compound_score > 0:
        return 2
    elif compound_score < 0:
        return 1
    else:
        return 0
