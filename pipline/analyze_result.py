import os

import pandas as pd


def read_result_data(fpath):
    df = pd.read_csv(str(fpath))
    return df


threshold_list = [
    0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
    0.7,
    0.8, 0.9, 1
]
task_list = ["sst2",
             # "sample_1000", "train", "sample_no"
             ]
for task in task_list:
    for threshold in threshold_list:
        root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        # 构建基于根目录的相对路径
        file_path = os.path.join(root_path, "analyze_data")
        result_path, similarity_path = '', ''
        for file_name in os.listdir(file_path):
            if file_name.endswith(f"{task}-{str(threshold)}-result.csv"):
                result_path = os.path.join(file_path, file_name)
            if file_name.endswith(f"{task}-{str(threshold)}-similarity.csv"):
                similarity_path = os.path.join(file_path, file_name)
        print(threshold, result_path, similarity_path)
        if os.path.exists(result_path) and os.path.exists(similarity_path):
            result_df = read_result_data(result_path)
            similarity_df = read_result_data(similarity_path)
            domain_counts = {}
            for index, row in result_df.iterrows():
                model_name = row["models"]
                model_res = []
                for column_name in result_df.columns:
                    if "MR" in column_name:
                        model_res.append(row[column_name])
                domain_counts[model_name] = model_res
            domain_counts["counts"] = list(similarity_df["counts"])
            average_value = 0.0
            for key, value in domain_counts.items():
                if key != "counts":
                    counts_values = domain_counts['counts']
                    result = sum(res * count for res, count in zip(value, counts_values))
                    result = result / sum(counts_values) / 100.0
                    domain_counts[key] = result
                    average_value += result
            print(f"{task}-{threshold}的平均情况为{average_value / 4.0}")
            print(domain_counts)
