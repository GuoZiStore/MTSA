import os
import time
from datetime import datetime

import bert_score
import numpy as np
import pandas as pd
from fast_bleu import SelfBLEU

from implement.data_generate import read_generate_data


def self_bleu_n(sentences, n):
    weights = {
        k: 1 / k * np.ones(k) for k in range(2, n + 1)
    }
    return np.mean(SelfBLEU([text.split() for text in sentences],
                            weights).get_score()[n])


def calculate_similarity(cand, ref):
    p, r, f1 = bert_score.score(cand, ref, lang="en")
    p_value = p.mean()
    return p_value


def mr_similar(task, rules, text_index, threshold):
    result = pd.DataFrame()
    result["MRs"] = rules
    similarity = []
    counts = []
    for mr in rules:
        run_time = 0
        start = time.time()
        try:
            filename = f"{mr}_test.csv"
            data = read_generate_data(task=task, filename=filename, threshold=threshold)
            count = len(data)
            print(f"正在计算{mr}生成数据相似度,文件名{filename},数据条数{count}")
            if len(data) > 1000:
                data = data.sample(n=1000, random_state=1)
            average_similarity = calculate_similarity(data[text_index].tolist(), data[text_index + 1].tolist())
            run_time += (time.time() - start)
            print(f"{mr}生成的数据平均相似度为:{average_similarity},运行时间:{run_time}")
            counts.append(count)
            similarity.append(average_similarity)
        except Exception as e:
            print(e)
            continue
    result["counts"] = counts
    result["similarity"] = similarity
    now = datetime.now()
    # 将日期转换为字符串
    current_date_str = now.strftime("%Y-%m-%d")
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_path, "result", f"{current_date_str}-{task}-{threshold}-similarity.csv")
    result.to_csv(file_path)


def mr_self_bleu(task, rules, text_index, threshold):
    result = pd.DataFrame()
    result["MRs"] = rules
    similarity = []
    for mr in rules:
        run_time = 0
        start = time.time()
        try:
            filename = f"{mr}_test.csv"
            data = read_generate_data(task=task, filename=filename, threshold=threshold)
            count = len(data)
            print(f"正在计算{mr}生成数据相似度,文件名{filename},数据条数{count}")
            if len(data) > 1000:
                data = data.sample(n=1000, random_state=1)
            average_similarity = self_bleu_n(data[text_index + 1].tolist(), 4)
            run_time += (time.time() - start)
            print(f"{mr}生成的数据self-bleu为:{average_similarity},运行时间:{run_time}")
            similarity.append(average_similarity)
        except Exception as e:
            print(e)
            continue
    result["similarity"] = similarity
    now = datetime.now()
    # 将日期转换为字符串
    current_date_str = now.strftime("%Y-%m-%d")
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    file_path = os.path.join(root_path, "result", f"{current_date_str}-{task}-{threshold}-similarity.csv")
    result.to_csv(file_path)
