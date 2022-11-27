import sys
from datasets import load_dataset

from multiprocessing.pool import ThreadPool

target_datasets = [
    ['wikitext', 'wikitext-103-raw-v1'],
    ['wikitext', 'wikitext-103-v1'],
    ['wikitext', 'wikitext-2-raw-v1'],
    ['wikitext', 'wikitext-2-v1'],

    ['openwebtext'],
    ['super_glue', 'boolq'],
    ['super_glue', 'cb'],
    ['super_glue', 'rte'],
    ['paws', 'labeled_final'],
    ['imdb'],
    ['snli'],
    ['scitail', 'snli_format'],
    ['SetFit/mrpc'],
    ['trec'],
    ['yelp_polarity'],
    ['iwslt2017', 'iwslt2017-ro-nl'],
    ['iwslt2017', 'iwslt2017-en-nl'],
    ['linxinyuan/cola'],
    ['sst2'],
    ['SetFit/stsb'],
    ['SetFit/qqp'],
    ['SetFit/mnli'],
    ['SetFit/qnli'],
    ['SetFit/rte'],
    ['SetFit/wnli'],
    ['social_i_qa'],
    ['cosmos_qa'],
    ['winogrande', 'winogrande_l'],
    ['hellaswag'],
    ['commonsense_qa'],
    ['sick'],
    ['wmt16', 'ro-en'],
    ['wmt16', 'cs-en'],
    ['wmt16', 'fi-en'],
    ['wmt14', 'hi-en'],
]

def download(dataset):
    try:
        if len(dataset) == 1:
            dataset0 = dataset[0]
            print("downloading " + dataset0)
            load_dataset(dataset0)
            print(dataset0 + " download done")
        else:
            dataset0, dataset1 = dataset
            print("downloading " + dataset0 + '+' + dataset1)
            # import pdb
            # pdb.set_trace()
            load_dataset(dataset0, dataset1)
            print(dataset0 + '+' + dataset1 + " download done")
    except:
        print(sys.exc_info())
    print("\n" * 5)

# p = ThreadPool(30)
# p.map(download, target_datasets)

# p = ThreadPool(30)
# p.map(download, target_datasets)

# p = ThreadPool(30)
# p.map(download, target_datasets)

# print("all done!")


print("check start!")
for dataset in target_datasets:
    while True:
        if len(dataset) == 1:
            dataset0 = dataset[0]
            print("downloading " + dataset0)
            load_dataset(dataset0)
            print(dataset0 + " download done")
        else:
            dataset0, dataset1 = dataset
            print("downloading " + dataset0 + '+' + dataset1)
            # import pdb
            # pdb.set_trace()
            load_dataset(dataset0, dataset1)
            print(dataset0 + '+' + dataset1 + " download done")
        break
    # except:
    #     print(sys.exc_info())
    #     exit(0)
    print("\n" * 5)
print("check end!")
