# Workflow of Ada-Ranker

## Introduction

Ada-Ranker could adaptively adjust the parameters of the ranking model based on the data distribution of the current candidate list, instead of using a parameter-frozen model for general service.

Below are the metric results on the ml-10m dataset comparing with original paper results.

|                         dataset                         |  metric  |  Base  |    End-to-End    | Finetune |
| :------------------------------------------------------: | :-------: | :-----: | :---------------: | :------: |
|                          ml-10m                          |    auc    | 0.92709 | **0.94533** | 0.94102 |
|                                                          | group_auc | 0.91598 |      0.92372      | 0.92041 |
| [paper](https://dl.acm.org/doi/abs/10.1145/3477495.3531931) | group_auc | 0.9187 |    **0.9279**    |          |

## Usage

To use Ada-Ranker, there are some optional dataset information files.

- item_emb_k.tsv.  The pretrained item embedding through Word2Vec. k means the embedding size.

You can directly generate the files needed by AdaRanker through running:

```bash
python Unirec/examples/preprocess/download_split_ml10m.py

bash UniRec/examples/preprocess/run_prepare_data-ml-10m-adaranker.sh
```

To run Ada-Ranker, there are two ways:

* The first is to train Ada-Ranker in an end-to-end way (only need to change `MY_DIR` and `ALL_DATA_ROOT`).
* The second is to train a base model first and then load pre-trained base model, and finetune all parameters in Ada-Ranker (`freeze=0`, set `SAVED_MODEL_PATH` to the path of pre-trained base model).

We provide a well-designed pipeline for simple usage of both ways (only need to change `pipeline`, 0 for base model training, 1 for ada-ranker end-to-end training, and 2 for base model training and ada-ranker fintuning).

```bash
bash UniRec/examples/ada-ranker/run_adaranker_pipeline_ml-10m-adaranker.sh
```
