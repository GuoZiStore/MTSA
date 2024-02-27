# -*- coding: utf-8 -*-
# @Time    : 2023/12/14 15:55
# @Author  : HuangHonghao
# @File    : data_generate.py
# @Description :
import os
import time
import pandas as pd
from tqdm import tqdm

from config.generation_rules import word_generation_rules, sentence_generate_rules, set_char_rules, set_word_rules
from implement.utils.char_utils import sentence_char_generate, sentence_level_generate, combine_sentence_generate


def read_data(task, filename):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_path, "asset", task, filename)
    if not os.path.exists(file_path):
        raise Exception(f"文件{filename}不存在")
    return pd.read_csv(str(file_path), header=None)


def read_generate_data(task, filename, threshold):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_path, "generate", task, str(threshold), filename)
    if not os.path.exists(file_path):
        raise Exception(f"文件{filename}不存在")
    return pd.read_csv(str(file_path), header=None)


def save_data(df, task, threshold, filename):
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_path, "generate", task, str(threshold), filename)
    parent_dir = os.path.join(root_path, "generate", task, str(threshold))
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    return df.to_csv(str(file_path), header=False, index=False)


def generate_data(filename, task, text_index, rules, threshold):
    tqdm.pandas()
    for mr, method in rules.items():
        org_data = read_data(task=task, filename=filename)
        run_time = 0
        start = time.time()
        if mr in word_generation_rules:
            new_data = org_data[text_index].progress_apply(sentence_char_generate, function=method, threshold=threshold)
        else:
            new_data = org_data[text_index].progress_apply(sentence_level_generate, function=method)
        org_data[text_index + 1] = new_data
        index_to_drop = org_data.index[org_data[text_index] == org_data[text_index + 1]]
        # 删除相同数据的行
        org_data = org_data.drop(index_to_drop)
        save_data(org_data, task=f"random_{task}", filename=f"{mr}_{filename}", threshold=threshold)
        run_time += (time.time() - start)
        print(f"蜕变关系{mr}生成数据{len(org_data)}条,运行时间:{run_time}")


def combine_generate_data(filename, task, threshold, text_index=1):
    run_time = 0
    start = time.time()
    tqdm.pandas()
    org_data = read_data(task=task, filename=filename)
    new_data = org_data[text_index].progress_apply(combine_sentence_generate, threshold=threshold,
                                                   char_rules=set_char_rules, word_rules=set_word_rules)
    org_data[text_index + 1] = new_data
    index_to_drop = org_data.index[org_data[text_index] == org_data[text_index + 1]]
    # 删除相同数据的行
    org_data = org_data.drop(index_to_drop)
    save_data(org_data, task=task, filename=f"combine_test.csv", threshold=threshold)
    run_time += (time.time() - start)
    print(f"蜕变组合生成数据{len(org_data)}条,运行时间:{run_time}")


def run_generate(filename, task, threshold, text_index=1):
    generate_data(filename, task, text_index, word_generation_rules, threshold)
    # generate_data(filename, task, text_index, sentence_generate_rules, threshold)
