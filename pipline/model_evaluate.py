import time

from config.generation_rules import model_predict_mrs
from implement.data_generate import run_generate
from implement.similar_util import mr_similar
from pipeline.sst2_test import model_predict
from train.mr_performance import mr_evaluate

# threshold_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
threshold_list = [0.7]
for threshold in threshold_list:
    print(f"正在处理阈值为{threshold}的情况")
    # 记录开始时间
    start_time = time.time()
    # 生成衍生测试用例
    # run_generate(task="sst2", filename="test.csv", threshold=threshold)
    # 预测原始用例和衍生用例结果
    # model_predict(run_type=1, task="sst2_train", threshold=threshold)
    # model_predict(run_type=2, task="sst2_train", threshold=threshold)
    # model_predict(run_type=3, task="sst2_train", threshold=threshold)
    # 生成数据相似度计算
    mr_similar(task="sst2", rules=model_predict_mrs, text_index=1, threshold=threshold)
    # 评估蜕变关系结果
    # mr_evaluate(model_predict_mrs, "sst2_train", threshold)
    # 记录结束时间
    end_time = time.time()
    # 计算经过的时间
    elapsed_time = end_time - start_time
    # 将时间转换为小时、分钟和秒
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"运行时间:{int(hours)}小时,{int(minutes)}分钟,{int(seconds)}秒")
