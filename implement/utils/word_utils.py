# -*- coding: utf-8 -*-
# @Time    : 2023/11/22 20:43
# @Author  : HuangHonghao
# @File    : word_utils.py
# @Description :
import random
import re
from functools import lru_cache

import Levenshtein
from nltk.stem import WordNetLemmatizer

from config import glove_model
from config.emoji_rules import word_emoji_dict
from config.word_rules import sample_typos_slang, sample_acronyms, sample_abbr
from implement.utils import cmu


def random_split_word(word):
    """
    随机单词分词
    @param word: 需要进行分词的单词
    @return: 分词后的单词
    """
    if len(word) <= 2:
        # 如果单词长度小于等于2，无法在中间插入空格
        return word
    else:
        # 随机选择一个位置，避免在开头和结尾插入空格
        split_position = random.randint(1, len(word) - 2)
        return word[:split_position] + ' ' + word[split_position:]


# 单词大小写转换
def word_to_uppercase(word):
    """
    单词大小写转换
    @param word:
    @return: 转换为大写后的单词
    """
    return word.upper()


def replace_similar_word(word):
    """
    同义词替换
    @param word: 需要进行替换的单词
    @return: 替换之后的同义词
    """
    try:
        result = glove_model.most_similar(word, topn=1)
        if not result:
            return word
        return result[0][0]
    except KeyError:
        return word


def word2emoji_replace(word):
    """
    单词表情符号替换
    @param word: 待替换的句子
    @return: 替换后的新句子
    """
    materialize = WordNetLemmatizer()
    word_emoji = {}
    lemma = materialize.lemmatize(word.lower())  # 词形还原
    emojis = word_emoji_dict.get(lemma)
    emoji = word
    if emojis:
        emoji = random.choice(emojis)
        word_emoji[word] = emoji
    return emoji


# def word2acronyms(sentence):
#     """
#     单词缩写替换
#     @param sentence: 待替换的句子
#     @return: 替换后的新句子
#     """
#     current_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
#     json_path = os.path.join(current_path, "config", "slang.json")
#     with open(json_path, 'r') as file:
#         slang = json.load(file)
#     slang_pattern = re.compile(
#         r'(?<!\w)(' + '|'.join(re.escape(key) for key in slang.keys()) + r')(?!\w)')
#     sentence = slang_pattern.sub(lambda x: slang[x.group()], sentence)
#     return sentence


def word2acronyms(sentence):
    """
    单词缩写替换
    @param sentence: 待替换的句子
    @return: 替换后的新句子
    """
    sample_typos_slang_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_typos_slang.keys()) + r')(?!\w)')
    sample_acronyms_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_acronyms.keys()) + r')(?!\w)')
    sample_abbr_pattern = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(key) for key in sample_abbr.keys()) + r')(?!\w)')

    sentence = sample_typos_slang_pattern.sub(lambda x: sample_typos_slang[x.group()], sentence)
    sentence = sample_acronyms_pattern.sub(lambda x: sample_acronyms[x.group()], sentence)
    sentence = sample_abbr_pattern.sub(lambda x: sample_abbr[x.group()], sentence)
    return sentence


@lru_cache(maxsize=None)
def get_pronunciations(word):
    """获取单词的音素表示
    @param word:
    @return:
    """
    word = word.lower()
    if word in cmu:
        return cmu[word]
    else:
        return []


def phonetic_similarity(word1, word2):
    """ 计算音素表示之间的相似度
    @param word1: 单词1
    @param word2: 单词2
    @return:
    """
    pronunciations1 = get_pronunciations(word1)
    pronunciations2 = get_pronunciations(word2)
    max_similarity = 0
    if len(pronunciations1) == 0 or len(pronunciations2) == 0:
        return max_similarity
    for p1 in pronunciations1:
        for p2 in pronunciations2:
            similarity = Levenshtein.ratio(''.join(p1), ''.join(p2))
            max_similarity = max(max_similarity, similarity)
    return max_similarity


def find_similar_words(input_word):
    """ 查找发音相似度大于阈值的单词
    @param input_word: 输入单词
    @return: 相似度>0.8的同音单词
    """
    similar_words = []
    threshold = 0.8
    for word in cmu.keys():
        similarity = phonetic_similarity(input_word, word)
        if similarity > threshold and input_word.lower() != word.lower():
            similar_words.append((word, similarity))
    similar_words.sort(key=lambda x: x[1], reverse=True)
    if len(similar_words) > 0:
        return str(similar_words[0][0])
    return input_word
