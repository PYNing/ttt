import sys

target_datasets = [
    'superglue-boolq',
    'superglue-cb',
    'superglue-rte',
    'paws',
    'imdb',
    'snli',
    'scitail',
    'mrpc',
    'trec',
    'yelp_polarity',
    'wmt16-ro-en',
    'wmt14-hi-en',
    'wmt16-en-ro',
    'wmt16-ro-en',
    'wmt16-en-cs',
    'iwslt2017-ro-nl',
    'iwslt2017-en-nl',
    'cola',
    'sst2',
    'stsb',
    'qqp',
    'mnli',
    'qnli',
    'rte',
    'wnli',
    'wmt16-en-fi',
    'social_i_qa',
    'cosmos_qa',
    'winogrande',
    'hellaswag',
    'commonsense_qa',
    'sick'
]

# try:
#     from datasets import load_dataset
# except:
#     print(sys.exc_info())


from tasks import  AutoTask

for dataset in target_datasets:
    try:
        print("downloading " + dataset)
        AutoTask.get(dataset)
        print(dataset + " download done")
    except:
        print(sys.exc_info())
        print("download " + dataset + " failed")
