import pandas as pd

from models.hub_config import sst2_models_list
from train.data_util import calculate_test_accuracy

tasks = ["sst2_train", "sst2"]
filename = "test.csv"
accuracies = {"sst2_train": [], "sst2": [], }
models_list = []
for model_name, model_path, model_type in sst2_models_list:
    models_list.append(model_name)
    for task in tasks:
        test_accuracy = calculate_test_accuracy(task, model_name, filename=filename)
        accuracies[task].append(test_accuracy)
df = pd.DataFrame(data={
    "model": models_list,
    "sst2_train": accuracies["sst2_train"],
    "sst2": accuracies["sst2"],
})
print(df)
