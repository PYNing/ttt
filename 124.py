small_datasets_without_all_splits = ["cola", "wnli", "rte", "trec", "superglue-cb", "sick",
                                     "mrpc", "stsb", "imdb", "commonsense_qa", "superglue-boolq"]
large_data_without_all_splits = ["yelp_polarity", "qqp", "qnli",
                                 "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "sst2"]

target_datasets = small_datasets_without_all_splits + large_data_without_all_splits

from datasets import load_dataset

for dataset in target_datasets:
    print("downloading" + dataset)
    load_dataset(dataset)
    print(dataset + "download done")