# UniRec

## Introduction

UniRec is an easy-to-use, lightweight, and scalable implementation of recommender systems. Its primary objective is to enable users to swiftly construct a comprehensive ecosystem of recommenders using a minimal set of robust and practical recommendation models. These models are designed to deliver scalable and competitive performance, encompassing a majority of real-world recommendation scenarios.


It is important to note that this goal differs from those of other well-known public libraries, such as Recommender and RecBole, which include missions of providing an extensive range of recommendation algorithms or offering various datasets.


The term "Uni-" carries several implications:

- Unit: Our aim is to employ a minimal set of models to facilitate the recommendation service onboarding process across most real-world scenarios. By maintaining a lightweight and extensible architecture, users can effortlessly modify and incorporate customized models into UniRec, catering to their specific future requirements.


- United: In contrast to the Natural Language Processing (NLP) domain, it is challenging to rely on a single model to serve end-to-end business applications in recommender systems. It is desirable that various modules or stages (such as retrieval and ranking) within a recommender system are not isolated and trained independently but are closely interconnected.


- Unified: While we acknowledge that model parameters cannot be unified, we believe there is potential to unify model structures. Consequently, we are exploring the possibility of utilizing a unified Transformer structure to serve different modules within recommender systems.


- Universal: We aspire for UniRec to support a wide range of recommendation scenarios, including gaming, music, movies, ads, and e-commerce, using a universal data model.



## Installation 


### Installation from PyPI

1. Ensure that [PyTorch](https://pytorch.org/get-started/previous-versions/) with CUDA supported (version 1.10.0-1.13.1) is installed:


    ```shell
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

    python -c "import torch; print(torch.__version__)"
    ```

2. Install `unirec` with pip:

    ```shell
    pip install unirec
    ```

### Installation from Wheel Locally

1. Ensure that [PyTorch](https://pytorch.org/get-started/previous-versions/) with CUDA supported (version 1.10.0-1.13.1) is installed:


    ```shell
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

    python -c "import torch; print(torch.__version__)"
    ```

2. Clone Git Repo

    ```shell
    git clone https://github.com/microsoft/UniRec.git
    ```

3. Build

    ```shell
    cd UniRec
    pip install --user --upgrade setuptools wheel twine
    python setup.py sdist bdist_wheel
    ```
    After building, the wheel package could be found in `UniRec/dist`.

4. Install

    ```shell
    pip install dist/unirec-*.whl 
    ```
    The specific package name could be find in `UniRec/dist`.

    Check if `unirec` is installed sucessfully:

    ```shell
    python -c "from unirec.utils import general; print(general.get_local_time_str())"
    ```


## Algorithms

| Algorithm | Type | Paper | Code |
|-----------|------|-------|------|
| MF |  Collaborative Filtering | [BPR](https://dl.acm.org/doi/10.5555/1795114.1795167) | [unirec/model/cf/mf.py](./unirec/model/cf/mf.py) |
| UserCF | Collaborative Filtering | - | [unirec/model/cf/usercf.py](./unirec/model/cf/usercf.py) |
| SLIM | Collaborative Filtering | [SLIM](https://ieeexplore.ieee.org/document/6137254) | [unirec/model/cf/slim.py](./unirec/model/cf/slim.py) |
| AdmmSLIM | Collaborative Filtering | [ADMMSLIM](https://dl.acm.org/doi/10.1145/3336191.3371774) | [unirec/model/cf/admmslim.py](./unirec/model/cf/admmslim.py) |
| SAR | Collaborative Filtering | [ItemCF](https://dl.acm.org/doi/10.1145/371920.372071), [SAR](https://github.com/recommenders-team/recommenders/blob/main/examples/02_model_collaborative_filtering/sar_deep_dive.ipynb) | [unirec/model/cf/sar.py](./unirec/model/cf/sar.py) |
| EASE | Collaborative Filtering | [EASE](https://dl.acm.org/doi/abs/10.1145/3308558.3313710) | [unirec/model/cf/ease.py](./unirec/model/cf/ease.py) |
| MultiVAE | Collaborative Filtering | [MultiVAE](https://dl.acm.org/doi/10.1145/3178876.3186150) | [unirec/model/cf/multivae.py](./unirec/model/cf/multivae.py) |
| SVDPlusPlus | Sequential Model | [SVD++](https://dl.acm.org/doi/10.1145/1644873.1644874) | [unirec/model/sequential/svdplusplus.py](./unirec/model/sequential/svdplusplus.py) |
| AvgHist | Sequential Model | - | [unirec/model/sequential/avghist.py](./unirec/model/sequential/avghist.py) |
| AttHist | Sequential Model | - | [unirec/model/sequential/atthist.py](./unirec/model/sequential/atthist.py) |
| GRU4Rec | Sequential Model | [GRU4Rec](https://dl.acm.org/doi/10.1145/2988450.2988452)  | [unirec/model/sequential/gru4rec.py](./unirec/model/sequential/gru4rec.py) |
| SASRec | Sequential Model | [SASRec](https://ieeexplore.ieee.org/abstract/document/8594844)  | [unirec/model/sequential/sasrec.py](./unirec/model/sequential/sasrec.py) |
| ConvFormer | Sequential Model | [ConvFormer](https://arxiv.org/abs/2308.02925)  | [unirec/model/sequential/convformer.py](./unirec/model/sequential/convformer.py) |
| FastConvFormer | Sequential Model | [ConvFormer](https://arxiv.org/abs/2308.02925) | [unirec/model/sequential/fastconvformer.py](./unirec/model/sequential/fastconvformer.py) |
| FM | Ranking Model | [Factorization Machine](https://ieeexplore.ieee.org/document/5694074)  | [unirec/model/rank/fm.py](./unirec/model/rank/fm.py) |
| BST | Ranking Model | [Behavior sequence transformer](https://dl.acm.org/doi/10.1145/3326937.3341261) | [unirec/model/rank/bst.py](./unirec/model/rank/bst.py) |


## Examples

To go through all the examples listed below, we provide a [script](./examples/preprocess/download_split_ml100k.py) for downloading and split for [ml-100k](https://grouplens.org/datasets/movielens/100k/) dataset. Run:

```shell
python download_split_ml100k.py
```

The files for the raw dataset would be saved in your home dir: `~/.unirec/dataset/ml-100k`

Next, it is essential to convert the raw dataset into a format compatible with UniRec. Use the [script](./examples/preprocess/preprocess_ml100k.sh) to process and save the files in `UniRec/data/ml-100k`.


```shell
cd examples/preprocess
bash preprocess_ml100k.sh
```


### General Training
To train an existing model in UniRec, for instance, training SASRec with ml-100k dataset, refer to the script provided in [examples/training/train_ml100k.sh](./examples/training/train_seq_ml100k.sh).


### Multi-GPU Training
UniRec supports multi-GPU training with the integration of [Accelerate](https://huggingface.co/docs/accelerate). An example script is available at [examples/training/multi_gpu_train_ml100k.sh](./examples/training/multi_gpu_train_ml100k.sh). The key arguments in the script could be found in line 3-12 in the script:

```shell
GPU_INDICES="0,1" # e.g. "0,1"

# Specify the number of nodes to use (one node may have multiple GPUs)
NUM_NODES=1

# Specify the number of processes in each node (the number should equal the number of GPU_INDICES)
NPROC_PER_NODE=2
```

For more details about the launching command, please refer to [Accelerate Docs](https://huggingface.co/docs/accelerate/basic_tutorials/launch).

### Hyperparameter Tuning with wandb

UniRec supports hyperparameter tuning (or hyperparameter optimization, HPO) with the intergration of [WandB](https://wandb.ai). There are three major steps to start a wandb experiment.


 1. Compose a training script and enable `wandb`. An example is provided in [examples/training/train_ml100k_with_wandb.sh](./examples/training/train_ml100k_with_wandb.sh). The key arguments are:

     - `--use_wandb=1`: enable wandb in process
     - `--wandb_file=/path/to/configuration_file`: the configuration file for wandb, including command, metrics, method, and search space.
 2. Define sweep configuration. Write a YAML-format configuration file to set the command, monitor metrics, tuning method and search space.An example is available at [examples/training/wandb.yaml](./examples/training/wandb.yaml). For more details about the configuration file, refer to [WandB Docs](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration)
 3. Initialize sweeps and start sweep agents. To start an experiment with wandb, first, initialize a sweep controller for selecting hyperparameters and issuing intructions; then an agent would actually perform the runs. An example for launching wandb experiments is provided in [examples/training/wandb_start.sh](./examples/training/wandb_start.sh). Note that we offer a pipeline command in the script to start the agent automatically after sweep initialization. However, we recommend the simpler manual two-step process:

 ```shell
## Step 1. Initialize sweeps with CLI using configuration file. 
## For more details, please refer to https://docs.wandb.ai/guides/sweeps/initialize-sweeps

wandb sweep config.yaml

## Step 2. After `wandb sweep`, you would get a sweep id and the hint to use `sweep agent`, like:

## wandb: Creating sweep from: ./wandb.yaml
## wandb: Created sweep with ID: xxx
## wandb: View sweep at: https://wandb.ai/xxx/xxx/xxx/xxx
## wandb: Run sweep agent with: wandb agent xxx/xxx/xxx/xxx

wandb agent entity/project/sweep_ID
```

### Serving with C# and Java

UniRec supports C# and Java inference based on [ONNX](https://onnxruntime.ai/docs/) format. We provide inference for user embedding, item embedding, and user-item score.

For more details, please refer to [examples/serving/README](./examples/serving/README)



## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
