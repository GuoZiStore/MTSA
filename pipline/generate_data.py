from implement.data_generate import run_generate, combine_generate_data

# run_generate(task="sst2", filename="test.csv", threshold=0.8)
# run_generate(task="sst2", filename="train.csv", threshold=0.7)
combine_generate_data(task="sst2", filename="test.csv", threshold=0.8)
