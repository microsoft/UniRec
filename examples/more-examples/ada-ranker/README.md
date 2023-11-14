# Workflow of Ada-Ranker

## Introduction

Ada-Ranker could adaptively adjust the parameters of the ranking model based on the data distribution of the current candidate list, instead of using a parameter-frozen model for general service.

## Usage

To use Ada-Ranker, there are some optional dataset information files.

- item_emb_k.tsv.  The pretrained item embedding through Word2Vec. k means the embedding size.

You can directly generate the files needed by AdaRanker through running:

```bash
bash UniRec/more-examples/preprocess/run_prepare_data-ml-100k-adaranker.sh
```

To run Ada-Ranker, there are two ways:

* The first is to train Ada-Ranker in an end-to-end way (only need to change `MY_DIR` and `ALL_DATA_ROOT`).

```bash
bash UniRec/more-examples/ada-ranker/run_adaranker_ml-100k-adaranker.sh
```

* The second is to train a base model first and then load pre-trained base model, and finetune all parameters in Ada-Ranker (`freeze=0`, set `SAVED_MODEL_PATH` to the path of pre-trained base model).

```bash
bash UniRec/more-examples/ada-ranker/run_adaranker_base_ml-100k-adaranker.sh

bash UniRec/more-examples/ada-ranker/run_adaranker_finetune_ml-100k-adaranker.sh
```

In addition, there is a well-designed pipeline for simple usuage.

```bash
bash UniRec/more-examples/ada-ranker/run_adaranker_pipeline_ml-100k-adaranker.sh
```
