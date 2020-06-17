# Text as Neural Operator: Image Manipulation by Text Instruction

This is the code for the paper:

**<a href="about:blank">Text as Neural Operator: Image Manipulation by Text Instruction
</a>**
<br>
Tianhao Zhang, Hung-Yu Tseng, Lu Jiang, Weilong Yang, Honglak Lee, Irfan Essa
<br>


*Please note that this is not an officially supported Google product.* And *this is the reproduced, not the original code.*

If you find this code useful in your research then please cite

```
TODO: Citation
```

## Introduction
In this paper, we study a new task that allows users to edit an input image using
language instructions.

![Problem Overview](images/teaser.png)

The key idea is to treat language as neural operators to locally modify the image feature.
To this end, our model decomposes the generation process into finding where (spatial region)
and how (text operators) to apply modifications. We show that the proposed model performs
favorably against recent baselines on three datasets.

![Method](images/overview.png)

## Installation

Clone this repo and go to the cloned directory.

Please create a environment using python 3.7 and install dependencies by
```bash
pip install -r requirements.txt
```

To reproduce the results reported in the paper, you would need an V100 GPU.

## Download datasets and pretrained model
Processed datasets (Clevr and Abstract Scene) and pretrained models can be downloaded
from [here](https://storage.googleapis.com/bryanzhang-bucket/dataset_n_models.tar). Extract the tar:
```
tar -xvf dataset_n_models.tar -C ../
```

## Testing Using Pretrained Model

Once the dataset and the pretrained model are downloaded,

1. Generate images using the pretrained model.
    ```bash
    bash run_test.sh
    ```
    Please switch parameters in the script for different datasets.

2. The outputs are at `../output/`.

## Training

New models can be trained with the following commands.

1. Prepare dataset. Processed datasets can be downloaded from the link above.
If you are to use a new dataset, please follow the structure of the provided
datasets, which means you need paired data (input image, input text, output image)

2. Train.

```bash
# Pretraining
bash run_pretrain.sh

# Training
bash run_train.sh
```

There are many options you can specify. We provide parameters for Clevr and Abstract Scene datasets in the script.

Tensorboard logs are stored at `[../checkpoints_local/TIMGAN/tfboard]`.

## Testing

Testing is similar to testing pretrained models.

```bash
bash run_test.sh
```


## Code Structure

- `run_pretrain.sh`, `run_train.sh`, `run_test.sh`: bash scripts for pretraining, training and testing.
- `train_recon.py`, `train.py`, `test.py`: the entry point for pretraining, training and testing.
- `models/tim_gan.py`: creates the networks, and compute the losses.
- `models/networks.py`: defines the basic modules for the architecture.
- `options/`: options.
- `dataset/`: defines the class for loading the dataset.
