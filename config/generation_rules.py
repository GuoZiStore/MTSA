# -*- coding: utf-8 -*-
# @Time    : 2023/12/14 15:56
# @Author  : HuangHonghao
# @File    : generation_rules.py
# @Description :
from implement.utils.char_utils import replace_visual_similar_char, replace_visual_split_char, replace_vowels, \
    insert_noise, replace_mask_char, swap_char, replace_word_keyboard
from implement.utils.sentence_utils import sentence_to_uppercase
from implement.utils.word_utils import *

set_char_rules = {
    # 视觉替换
    "MR1": replace_visual_similar_char,
    # 视觉分词
    "MR2": replace_visual_split_char,
    # 随机重复
    "MR3": replace_vowels,
    # 噪声注入
    "MR4": insert_noise,
    # 字符遮罩
    "MR5": replace_mask_char,
    # 字符交换
    "MR6": swap_char,
}
set_word_rules = {
    # 同音字替换
    "MR7": find_similar_words,
    # 键盘错误
    "MR8": replace_word_keyboard,
    # 同义词替换
    "MR9": replace_similar_word,
    # 随机分词空格
    "MR11": random_split_word,
    # 大小写转换
    "MR12": word_to_uppercase,
    # 表情替换
    "MR13": word2emoji_replace,
}
# 操作单词的规则
word_generation_rules = {
    # 视觉替换
    "MR1": replace_visual_similar_char,
    # 视觉分词
    "MR2": replace_visual_split_char,
    # 随机重复
    "MR3": replace_vowels,
    # 噪声注入
    "MR4": insert_noise,
    # 字符遮罩
    "MR5": replace_mask_char,
    # 字符交换
    "MR6": swap_char,
    # 同音字替换
    "MR7": find_similar_words,
    # 键盘错误
    "MR8": replace_word_keyboard,
    # 同义词替换
    "MR9": replace_similar_word,
    # 随机分词空格
    "MR11": random_split_word,
    # 大小写转换
    "MR12": word_to_uppercase,
    # 表情替换
    "MR13": word2emoji_replace,
}
# 操作句子的规则
sentence_generate_rules = {
    # 句子大小写
    "MR14": sentence_to_uppercase,
    # 单词缩写替换
    "MR10": word2acronyms,
}
model_predict_mrs = [
    "MR1",
    "MR2",
    "MR3",
    "MR4",
    "MR5",
    "MR6",
    "MR7",
    "MR8",
    "MR9",
    "MR10",
    "MR11",
    "MR12",
    "MR13",
    "MR14",
    "combine",
]

am_model_predict_mrs = [
    "MR1",
    "MR2",
    "MR3",
    "MR4",
    "MR5",
    "MR6",
    "MR7",
    "MR8",
    "MR9",
    "MR10",
    "MR11",
    "MR12",
    "MR13",
    "MR14",
    "combine",
]
