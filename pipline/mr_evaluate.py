# 运行MR的原始结果
import os

import pandas as pd

from config.generation_rules import model_predict_mrs
from implement.data_generate import read_data
from implement.similar_util import mr_similar
from models.hub_config import sst2_models_list
from pipeline.sst2_test import read_predict_data, model_predict, retrain_model_predict
from train.data_util import calculate_test_accuracy
from train.mr_performance import mr_evaluate


def org_model_evaluate(task_list):
    for task in task_list:
        mr = "acc"
        test_df = read_data(task="sst2", filename="test.csv")
        test_texts = test_df[1].values
        retrain_model_predict(model_list=sst2_models_list, test_texts=test_texts, task=task, mr=mr)
    accuracies = {task: [] for task in task_list}
    model_name_list = ["distilbert-base-cased-SST-2", "albert-base-v2-SST-2", "bert-base-uncased-SST-2",
                       "roberta-base-SST-2",
                       ]
    models_list = []
    for model_name in model_name_list:
        models_list.append(model_name)
        for task in task_list:
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fpath = os.path.join(root_path, "predictions", "sst2", "acc", f"{model_name}.pkl")
            test_accuracy = calculate_test_accuracy(task="sst2", pred_path=fpath, filename="test.csv", label_index=0)
            accuracies[task].append(test_accuracy)
    df = pd.DataFrame(data={
        "model": models_list,
        **{task: accuracies[task] for task in task_list}
    })
    print(df)


def mr_model_evaluate(task_list, index, is_test, threshold=0.7, train=0):
    for task in task_list:
        for mr in model_predict_mrs:
            filename = f"{mr}_test.csv"
            if train:
                filename = f"{mr}_train.csv"
            _, test_df = read_predict_data(task=task, filename=filename, index=index, threshold=threshold)
            model_predict(model_list=sst2_models_list, test_texts=test_df, task=task, threshold=threshold, mr=mr,
                          is_test=is_test, train=train)


def mr_retrain_model_evaluate(task_list, index, is_test):
    threshold = 0.7
    for task in task_list:
        for mr in model_predict_mrs:
            filename = f"{mr}_test.csv"
            model_list = [
                ("distilbert-base-cased-SST-2", f"../retrain_models/{task}/distilbert-base-cased-SST-2", "distilbert"),
                ("albert-base-v2-SST-2", f"../retrain_models/{task}/albert-base-v2-SST-2", "albert"),
                ("bert-base-uncased-SST-2", f"../retrain_models/{task}/bert-base-uncased-SST-2", "bert"),
                ("roberta-base-SST-2", f"../retrain_models/{task}/roberta-base-SST-2", "roberta")
            ]
            _, test_df = read_predict_data(task="sst2", filename=filename, index=index, threshold=threshold)
            model_predict(model_list=model_list, test_texts=test_df, task=task, threshold=threshold, mr=mr,
                          is_test=is_test)


# 重新训练模型评估
def retrain_model_evaluate(task_list):
    for task in task_list:
        test_df = read_data(task="sst2", filename="test.csv")
        test_texts = test_df[1].values
        model_list = [
            ("distilbert-base-cased-SST-2", f"../retrain_models/{task}/distilbert-base-cased-SST-2", "distilbert"),
            ("albert-base-v2-SST-2", f"../retrain_models/{task}/albert-base-v2-SST-2", "albert"),
            ("bert-base-uncased-SST-2", f"../retrain_models/{task}/bert-base-uncased-SST-2", "bert"),
            ("roberta-base-SST-2", f"../retrain_models/{task}/roberta-base-SST-2", "roberta")
        ]
        retrain_model_predict(model_list=model_list, test_texts=test_texts, task="sst2", mr=task)
    accuracies = {task: [] for task in task_list}
    model_name_list = ["distilbert-base-cased-SST-2", "albert-base-v2-SST-2", "bert-base-uncased-SST-2",
                       "roberta-base-SST-2",
                       ]
    models_list = []
    for model_name in model_name_list:
        models_list.append(model_name)
        for task in task_list:
            root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            fpath = os.path.join(root_path, "predictions", "sst2", task, f"{model_name}.pkl")
            test_accuracy = calculate_test_accuracy(task="sst2", pred_path=fpath, filename="test.csv", label_index=0)
            accuracies[task].append(test_accuracy)
    df = pd.DataFrame(data={
        "model": models_list,
        **{task: accuracies[task] for task in task_list}
    })
    print(df)


mr_task_list = ["sst2"]
threshold_list = [
    # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    # 0.7,
    0.8,
    # 0.9, 1
]
for threshold in threshold_list:
    #     # 运行原始用例
    # mr_model_evaluate(task_list=mr_task_list, index=1, is_test="org", threshold=threshold, train=0)
    # #     # # 运行衍生用例
    # mr_model_evaluate(task_list=mr_task_list, index=2, is_test="test", threshold=threshold, train=0)
    mr_evaluate(rules=model_predict_mrs, task="sst2", threshold=threshold)
#     # 生成数据相似度计算
#     mr_similar(task="sst2", rules=model_predict_mrs, text_index=1, threshold=threshold)
# mr_similar(task="sst2", rules=model_predict_mrs, text_random_sst2index=1, threshold=threshold)
# 评估重新训练模型
# retrain_model_evaluate(task_list=["sample_1000", "sample_no"])
# retrain_model_evaluate(task_list=["only_error"])
# retrain_model_evaluate(task_list=["train_200"])
# mr_retrain_model_evaluate(task_list=["sample_1000", "sample_no"], index=1, is_test="org")
# mr_retrain_model_evaluate(task_list=["sample_1000", "sample_no"], index=2, is_test="test")
# for task in ["sample_1000", "sample_no"]:
#     mr_evaluate(model_predict_mrs, task, 0.7)
# mr_retrain_model_evaluate(task_list=["train_200"], index=1, is_test="org")
# mr_retrain_model_evaluate(task_list=["train_200"], index=2, is_test="test")
# for task in ["train_200"]:
#     mr_evaluate(model_predict_mrs, task, 0.7)
# org_model_evaluate(task_list=mr_task_list)
