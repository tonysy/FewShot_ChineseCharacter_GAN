# Chinese Character Generation with Few-shot Images

This project is design to generate personalized chinese character with few-shot images.

## Installation
```
# Install some packages
pip install dominate

# Install the project
python setup.py build develop  
```
## Algorithms
### Data Preparation
We collect 56 fonts from the Internet.

We first extract the images from the ttf with 
### Model Description
### Traning

#### Step-1: Train Font Classification Network
```
cd tools
bash tools/cat_embedding_train.sh
```
#### Step-2: Extract Category Embedding Features
```
bash tools/cat_embedding_train_feat_extraction.sh
```
#### Step-3: Train Category-aware Pix2Pix
```
bash tools/gan_model_version1.sh
```
### Inference
### Visualization


## Server
### Flask

## Front-end

### HTML
### PyQt

## Acknoledgement
