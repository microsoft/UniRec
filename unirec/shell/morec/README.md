# Workflow of MoRec

## Introduction

MoRec consists of a data sampler for unified objective modeling and a objective controller for objective balancing.

MoRec could be used as a plugin for various deeplearning based models. And validation data is required for tri-level optimization.


## Usage
To use MoRec, there are some required several dataset information files.

- item2price.tsv.  The price of item. The first column is item_id and the second column is price.
- item2category.tsv.  The category of item. The first column is item_id and the second column is category.

(Both the seprator should be `,`.)


To run MoRec, the first step is to pretrain a base model.

```bash
bash UniRec/unirec/shell/morec/run_base_model.sh
```

After the training procedure is done, we can run MoRec to finetune the model.

```bash
bash UniRec/unirec/shell/morec/run_morec.sh
```

In addition, there is a well-designed pipeline for simple usuage.

```bash
bash UniRec/unirec/shell/morec/run_pipeline.sh
```
