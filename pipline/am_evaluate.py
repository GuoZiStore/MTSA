import time

from config.generation_rules import model_predict_mrs, am_model_predict_mrs
from pipeline.sst2_test import nltk_mr_predict
from train.mr_performance import mr_nltk_evaluate

threshold_list = [0.7]
task_list = [
    "nltk",
    "baidu",
    "amazon",
    "azure",
    # "huawei"
]
for threshold in threshold_list:
    for task in task_list:
        print(f"正在处理任务{task}阈值为{threshold}的情况")
        # 记录开始时间
        start_time = time.time()
        # 预测原始用例和衍生用例结果
        # nltk_mr_predict(run_type=2, task=task, threshold=threshold)
        # nltk_mr_predict(run_type=3, task=task, threshold=threshold)
        # 评估蜕变关系结果
        mr_nltk_evaluate(am_model_predict_mrs, task, threshold)
        # 记录结束时间
        end_time = time.time()
        # 计算经过的时间
        elapsed_time = end_time - start_time
        # 将时间转换为小时、分钟和秒
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print(f"运行时间:{int(hours)}小时,{int(minutes)}分钟,{int(seconds)}秒")
