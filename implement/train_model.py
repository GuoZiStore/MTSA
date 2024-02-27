import os

import pandas as pd
from simpletransformers.classification import ClassificationModel
from simpletransformers.config.model_args import ClassificationArgs

from config.generation_rules import model_predict_mrs
from implement.data_generate import read_data
from pipeline.sst2_test import read_predict_data
from train.data_util import load_pickle


def read_train_data(f_path, f_name, threshold, label_index=0, text_index=1):
    """
    @param f_path: 读取文件路径
    @param f_name: 读取文件名
    @param threshold: 设置参数
    @param label_index: 类别标签列，默认 0
    @param text_index: 文本标签列，默认 1
    @return: 返回dataframe数据
    """
    root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # 构建基于根目录的相对路径
    file_path = os.path.join(root_path, "generate", f_path, str(threshold), f_name)
    df = pd.read_csv(str(file_path), header=None)
    train_df = pd.DataFrame(columns=['labels', 'text'])
    train_df["text"] = df[text_index + 1]
    train_df["labels"] = df[label_index]
    return train_df


def read_all_train_data(f_path, threshold):
    """
    @param f_path: 读取文件路径
    @param threshold: 设置参数
    @return: 返回dataframe数据
    """
    dfs = []
    for mr in model_predict_mrs:
        # 构建基于根目录的相对路径
        filename = f"{mr}_train.csv"
        df = read_train_data(f_path=f_path, f_name=filename, threshold=threshold)
        dfs.append(df)
    train_df = pd.concat(dfs, ignore_index=True)
    return train_df


def read_just_train_data(f_path):
    train_df = read_data(task=f_path, filename="train.csv")
    train_df = train_df.rename(columns={train_df.columns[0]: 'labels', train_df.columns[1]: 'text'})
    return train_df


def read_train_all_error_data(f_path, threshold, model_name):
    dfs = []
    train_df = read_data(task=f_path, filename="train.csv")
    train_df = train_df.rename(columns={train_df.columns[0]: 'labels', train_df.columns[1]: 'text'})
    dfs.append(train_df)
    for mr in model_predict_mrs:
        # 构建基于根目录的相对路径
        filename = f"{mr}_train.csv"
        df = read_train_data(f_path=f_path, f_name=filename, threshold=threshold)
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建基于根目录的相对路径
        org_path = os.path.join(root_path, "predictions", f_path, str(threshold), mr, "org", f"{model_name}_train.pkl")
        test_path = os.path.join(root_path, "predictions", f_path, str(threshold), mr, "test",
                                 f"{model_name}_train.pkl")
        org_pre = load_pickle(org_path)
        test_pre = load_pickle(test_path)
        res_df = pd.DataFrame({'org_pre': org_pre, 'test_pre': test_pre})
        train_df = pd.concat([df, res_df], axis=1)
        filtered_df = train_df[train_df['org_pre'] != train_df['test_pre']]
        # 删除 org_pre 和 test_pre 两列
        filtered_df = filtered_df.drop(['org_pre', 'test_pre'], axis=1)
        count = len(filtered_df)
        if count > 200:
            filtered_df = filtered_df.sample(n=200, random_state=1)
        dfs.append(filtered_df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def read_train_add_enhance_data(f_path, threshold, sample_count):
    dfs = []
    train_df = read_data(task=f_path, filename="train.csv")
    train_df = train_df.rename(columns={train_df.columns[0]: 'labels', train_df.columns[1]: 'text'})
    dfs.append(train_df)
    for mr in model_predict_mrs:
        # 构建基于根目录的相对路径
        filename = f"{mr}_train.csv"
        df = read_train_data(f_path=f_path, f_name=filename, threshold=threshold)
        count = len(df)
        if sample_count:
            if count > sample_count:
                df = df.sample(n=sample_count, random_state=1)
        dfs.append(df)
    combined_df = pd.concat(dfs, ignore_index=True)
    return combined_df


def train_model(model_name, model_type, num_labels, train_df, mr):
    import os

    # 禁用并行处理
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
    save_path = model_name.replace("../models/", "")
    model_args = ClassificationArgs()
    model_args.manual_seed = 42

    model_args.max_seq_length = 128
    model_args.train_batch_size = 2
    model_args.eval_batch_size = 2

    model_args.use_multiprocessing = False
    model_args.use_multiprocessing_for_evaluation = False
    model_args.multiprocessing_chunksize = 8
    model_args.dataloader_num_workers = 8

    model_args.num_train_epochs = 3
    model_args.save_model_every_epoch = False
    model_args.save_steps = -1
    model_args.evaluate_during_training = False

    model_args.output_dir = f"../retrain_models/{mr}/{save_path}"
    model_args.best_model_dir = f"../retrain_models/{mr}/{save_path}/best_model"
    model_args.tensorboard_dir = f"../retrain_models/runs/{mr}/{save_path}/testing"
    model_args.overwrite_output_dir = True
    print(model_args.output_dir)
    print(model_args.best_model_dir)
    model = ClassificationModel(
        model_type,
        model_name,
        num_labels=num_labels,
        args=model_args,
        cuda_device=0,
    )
    model.train_model(
        train_df,
        eval_df=None
    )
