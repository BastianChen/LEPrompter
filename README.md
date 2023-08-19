# High-Fidelity Lake Extraction via Two-Stage Prompt Enhancement: Establishing a Novel Baseline and Benchmark

<p align="center">
    <img src="./resources/overall_architecture_diagram.png">
</p>

Figure 1: Overview architecture of LEPrompter with three main modules. (a) A prompt dataset that contains prior information. (b) A prompt encoder that extracts strong prior prompt information features. (c) A lightweight decoder that fuses the prompt tokens from the prompt encoder and the image embedding from the Vision Image Encoder to generate the final lake mask.

The repository contains official PyTorch implementations of training and evaluation codes and pre-trained models for **LEPrompter**.

The code is based on [MMSegmentaion v0.30.0](https://github.com/open-mmlab/MMSegmentation/tree/v0.30.0).

## Installation

For install and data preparation, please refer to the guidelines in [MMSegmentation v0.30.0](https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0).

An example (works for me): ```CUDA 11.6``` and  ```pytorch 1.11.0``` 

```
pip install -U openmim
mim install mmcv-full
cd LEFormer && pip install -e . --user
```

## Pretrained Weights
Due to the size limitation of 50MB for the Supplementary Material, we are currently unable to provide the pretrained weights. After the paper has been accepted, we will make the download links for the pretrained weights available.


## Datasets Preparation

After the paper has been accepted, we will make the download links for the Surface Water dataset (SW dataset) and the Qinghai-Tibet Plateau Lake dataset (QTPL dataset) that we used available.

### Split Dataset
Alternatively, the datasets can be recreated to randomly split the datasets into training and testing sets, based on the original datasets.  

The original SW dataset is freely available for download [here](https://aistudio.baidu.com/aistudio/datasetdetail/75148).

The original QTPL dataset is freely available for download [here](http://www.ncdc.ac.cn/portal/metadata/b4d9fb27-ec93-433d-893a-2689379a3fc0).

Example: split ```Surface Water```:
```python
python tools/data_split.py --dataset_type sw --dataset_path /path/to/your/surface_water/train_data --save_path /path/to/save/dataset
```

Example: split ```Qinghai-Tibet Plateau Lake```:
```python
python tools/data_split.py --dataset_type qtpl --dataset_path /path/to/your/LakeWater --save_path /path/to/save/dataset
```
### Create Prompt Dataset

The structure of prompt datasets are aligned as follows:
```
SW or QTPL
├── annotations
│   ├── training 
│   └── validation 
├── binary_annotations
│   ├── training 
│   └── validation 
├── images  
│   ├── training 
│   └── validation 
└── prompts  
    └── training  
```

Example: create ```Surface Water Prompt Dataset```:
```python
python tools/gen_prompt_datasets.py --dataset_path /path/to/your/surface_water/
```

## Training

We use 1 GPU for training by default. Make sure you have modified the `data_root` variable in [prompt_sw_256x256.py](local_configs/_base_/datasets/prompt_sw_256x256.py) or [prompt_qtpl_256x256.py](local_configs/_base_/datasets/prompt_qtpl_256x256.py).    

Example: train ```LEPrompter``` on ```Surface Water Prompt Dataset```:

```python
python tools/train.py local_configs/leprompter/leprompter_256x256_sw_160k.py
```

## Evaluation
To evaluate the model. Make sure you have modified the `data_root` variable in [sw_256x256.py](configs/_base_/datasets/sw_256x256.py) or [qtpl_256x256.py](configs/_base_/datasets/qtpl_256x256.py).  

Example: evaluate ```LEPrompter``` on ```Surface Water Prompt Dataset```:

```python
python tools/test.py configs/leformer/leformer_256x256_sw_160k.py /path/to/your/pretrained_model --eval mIoU mFscore
```
## Supplement 
### User Study
