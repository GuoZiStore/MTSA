import os
import pickle
import time

from tqdm import tqdm

from api.aws_api import aws_api
from api.azure_api import azure_api
from api.baidu_api import baidu_api
from api.huawei_api import huawei_api
from api.tencent_api import tencent_api
from config.generation_rules import model_predict_mrs, am_model_predict_mrs
from implement.nltk_utils import nltk_predict
from train.data_util import read_generate_data, calculate_test_accuracy
from train.test_model import test_model


def read_predict_data(task, filename, index, threshold):
    test_labels, test_texts = read_generate_data(f_path=task, f_name=filename, text_index=index, threshold=threshold)
    return test_labels, test_texts


def save_predict_data(f_path, predict_df):
    parent_dir = "/".join(str(f_path).split('/')[:-1])
    print(f"父路径{parent_dir}")
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(f_path, 'wb') as f:
        pickle.dump(predict_df, f)


def model_predict(model_list, test_texts, task, threshold, mr, is_test, train=0):
    for model_name, model_path, model_type in model_list:
        predict_df = test_model(test_texts=test_texts, model_type=model_type, model_path=model_path)
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        save_name = f"{model_name}.pkl"
        if train:
            save_name = f"{model_name}_train.pkl"
        fpath = os.path.join(root_path, "predictions", task, str(threshold), mr, is_test, save_name)
        save_predict_data(f_path=fpath, predict_df=predict_df)


def retrain_model_predict(model_list, test_texts, task, mr):
    for model_name, model_path, model_type in model_list:
        predict_df = test_model(test_texts=test_texts, model_type=model_type, model_path=model_path)
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fpath = os.path.join(root_path, "predictions", task, mr, f"{model_name}.pkl")
        print(f"保存路径{fpath}")
        save_predict_data(f_path=fpath, predict_df=predict_df)


def nltk_mr_predict(run_type, task, threshold):
    # 运行MR的结果
    for mr in am_model_predict_mrs:
        run_time = 0
        start = time.time()
        filename = f"{mr}_test.csv"
        if run_type == 3:
            index = 2
            is_test = "test"
        else:
            index = 1
            is_test = "org"
        test_labels, test_texts = read_generate_data(f_path=task, f_name=filename, text_index=index,
                                                     threshold=threshold)
        pre_res = []
        for text in tqdm(test_texts, desc="情感分析", unit="句"):
            if task == "nltk":
                score = nltk_predict(text)
                pre_res.append(score)
            elif task == "baidu":
                result = baidu_api(text)
                score = result.sentiment
                pre_res.append(score)
            elif task == "amazon":
                score = aws_api(text)
                pre_res.append(score)
            elif task == "huawei":
                score = huawei_api(text)
                pre_res.append(score)
            elif task == "azure":
                max_retry = 3  # 设置最大重试次数
                retry_count = 0
                while retry_count < max_retry:
                    try:
                        score = azure_api(text)
                        pre_res.append(score)
                    except Exception as e:
                        print(f"An error occurred: {e}")
                        retry_count += 1
                        print(f"Retrying... (retry count: {retry_count})")
                    else:
                        # 如果没有发生异常，则跳出循环
                        break
                else:
                    # 如果达到最大重试次数仍然失败，可以在这里处理
                    print("Max retry count reached. Moving to the next data.")
                    break
        print(f"需要处理{len(test_texts)}条，结果{len(pre_res)}条")
        if len(test_texts) != len(pre_res):
            print(f"{mr}数据与结果不相等")
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        fpath = os.path.join(root_path, "predictions", task, str(threshold), mr, is_test, f"{task}.pkl")
        parent_dir = os.path.join(root_path, "predictions", task, str(threshold), mr, is_test)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(fpath, 'wb') as f:
            pickle.dump(pre_res, f)
        run_time += (time.time() - start)
        print(f"{mr}运行时间:{run_time}")
