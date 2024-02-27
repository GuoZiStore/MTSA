import random
import re

from config.char_rules import char_keyboard_keys, char_keyboard_error, char_split, char_replace, char_combine
from implement.nltk_utils import get_disturbance_list, count_words


def sentence_char_generate(sentence, function, threshold):
    replaced_sentence = sentence
    # words = get_sentiment_words(sentence)
    words = get_disturbance_list(sentence, threshold)
    if len(words) == 0:
        return sentence
    n = len(words)
    words = random.sample(count_words(sentence), n)
    for word in words:
        replaced_word = function(word)
        replaced_sentence = re.sub(r'\b' + re.escape(word) + r'\b', replaced_word, replaced_sentence)
    return replaced_sentence


def get_replace_sentence(sentence, word_list, rules_dict):
    for word in word_list:
        function_key = random.choice(list(rules_dict.keys()))
        function = rules_dict[function_key]
        replaced_word = function(word)
        sentence = re.sub(r'\b' + re.escape(word) + r'\b', replaced_word, sentence)
    return sentence


def combine_sentence_generate(sentence, threshold, char_rules, word_rules):
    replaced_sentence = sentence
    words = get_disturbance_list(sentence, threshold)
    if len(words) == 0 or len(words) < 2:
        return sentence
    random.shuffle(words)
    array_length = len(words)
    half_length = array_length // 2
    char_words = words[:half_length]
    word_words = words[half_length:]
    replaced_sentence = get_replace_sentence(replaced_sentence, char_words, char_rules)
    replaced_sentence = get_replace_sentence(replaced_sentence, word_words, word_rules)
    replaced_sentence.upper()
    return replaced_sentence


def sentence_level_generate(sentence, function):
    return function(sentence)


def replace_visual_similar_char(word):
    """
    视觉相似字符：替换视觉相似字符
    @param word: 需要操作的单词
    @return: 替换后的单词
    """
    replaced_word = ''.join(char_replace.get(char, char) for char in word)
    return replaced_word


def replace_visual_split_char(word):
    """
    视觉分词字符：替换视觉分词相似字符
    @param word: 需要操作的单词
    @return: 替换后的单词
    """
    replaced_word = ''.join(char_split.get(char, char) for char in word)
    return replaced_word


def replace_visual_combine_char(word):
    """
    视觉组合字符：替换视觉组合字符
    @param word: 需要操作的单词
    @return: 替换后的单词
    """
    replaced_word = ''.join(char_combine.get(char, char) for char in word)
    return replaced_word


def replace_vowels(word):
    """
    随机重复元音字母
    @param word: 需要操作的单词
    @return: 替换后的单词
    """
    vowels = [char for char in word if char in 'aeiou']
    if vowels:
        chosen_vowel = random.choice(vowels)
        replaced_word = word.replace(chosen_vowel, chosen_vowel * random.randint(2, 4), 1)
        return replaced_word
    else:
        return word


def insert_noise(word):
    """
    噪声注入：给定一个单词，向其随机插入一个'*'字符
    @param word: 需要插入噪声的单词
    @return: 插入噪声后的单词
    """
    # 如果单词长度小于等于2，则不插入噪声
    if len(word) <= 2:
        return word
    # 随机选择插入的位置，避免在首尾插入
    position = random.randint(1, len(word) - 1)
    # 在选择的位置插入噪声 '*'
    word_with_noise = word[:position] + '*' + word[position:]
    return word_with_noise


def replace_mask_char(word):
    """
    字符遮罩：替换字符函数
    @param word: 需要替换的单词
    @return: 替换后的单词
    """
    # 如果单词长度小于等于3，则不替代
    if len(word) <= 3:
        return word
    # 随机选择替代的位置，避免在首尾替代
    position = random.randint(1, len(word) - 2)
    # 在选择的位置用 '#' 替代
    word_with_replacement = word[:position] + '#' + word[position + 1:]
    return word_with_replacement


def swap_char(word):
    """
    字符交换：交换单词中某个位置及其相邻位置的字符
    @param word: 需要交换字符的单词
    @return: 交换字符后的单词
    """
    # 如果单词长度小于等于2，则不交换
    if len(word) <= 2:
        return word
    # 随机选择一个位置，该位置及其相邻的位置进行交换
    position = random.randint(0, len(word) - 2)
    # 将选择的位置与其相邻位置的字符交换
    word_list = list(word)
    word_list[position], word_list[position + 1] = word_list[position + 1], word_list[position]
    word_with_swap = ''.join(word_list)
    return word_with_swap


def replace_word_keyboard(word):
    """
    键盘错误：替换word中的字符为键盘上可能的替代字符

    @param word: 待替换的字符串
    @return: 替换后的字符串
    """
    valid_chars = [char for char in word if char in char_keyboard_keys]
    if not valid_chars:
        return word
    replace_1char = random.choice(valid_chars)
    replace_index = word.index(replace_1char)
    replacement = random.choice(char_keyboard_error[replace_1char])
    replaced_list = list(word)
    replaced_list[replace_index] = replacement
    replaced_word = ''.join(replaced_list)
    return replaced_word
