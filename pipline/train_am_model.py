import os
import time

from implement.train_model import train_model, read_train_add_enhance_data, read_train_all_error_data, \
    read_just_train_data
from models.hub_config import sst2_models_list

# 禁用并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def train_enhance_model(f_path, threshold, sample_count, mr, num_labels):
    if sample_count != -1 and sample_count != -2:
        train_df = read_train_add_enhance_data(f_path=f_path, threshold=threshold, sample_count=sample_count)
        print(train_df.shape)
    elif sample_count == -2:
        train_df = read_just_train_data(f_path="sst2")
        print(train_df.shape)
    else:
        train_df = None
    for model_name, model_path, model_type in sst2_models_list:
        if sample_count == -1:
            train_df = read_train_all_error_data(f_path=f_path, threshold=threshold, model_name=model_name)
            print(train_df.shape)
        run_time = 0
        start = time.time()
        if train_df is not None:
            train_model(model_name=model_path, model_type=model_type, num_labels=num_labels, train_df=train_df, mr=mr)
        run_time += (time.time() - start)
        print(f"{model_name}运行时间:{run_time}")


# train_enhance_model(f_path="sst2", threshold=0.7, sample_count=0, mr="sample_no", num_labels=2)
# train_enhance_model(f_path="sst2", threshold=0.7, sample_count=1000, mr="sample_1000", num_labels=2)
# train_enhance_model(f_path="sst2", threshold=0.7, sample_count=-1, mr="only_error", num_labels=2)
train_enhance_model(f_path="sst2", threshold=0.7, sample_count=-1, mr="train_200", num_labels=2)
