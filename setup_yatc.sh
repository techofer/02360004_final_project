#!/bin/bash

PRETRAINED_MODEL_NAME=./output_dir/pretrained-model.pth
FINE_TUNING_DATASET_ZIP_NAME=data.zip
FINE_TUNING_DATASET_DIR=YaTC_datasets

sudo apt-get install git unzip -y

pip install pip -U
pip install gdown timm==0.3.2 torch==1.8.1 numpy==1.19.5 scikit-learn==0.24.2 tensorboard scikit-image matplotlib || exit

if [ ! -d YaTC ]; then
  git clone https://github.com/NSSL-SJTU/YaTC.git || exit
fi

cd YaTC || exit

mkdir -p output_dir

if [ ! -f $PRETRAINED_MODEL_NAME ]; then
  gdown https://drive.google.com/uc?id=1wWmZN87NgwujSd2-o5nm3HaQUIzWlv16 -O $PRETRAINED_MODEL_NAME  || exit

fi
if [ ! -f $FINE_TUNING_DATASET_ZIP_NAME ]; then
  gdown https://drive.google.com/uc?id=1znKQpZ704Bh4EkaHUBJwztYgflFXPnHI -O $FINE_TUNING_DATASET_ZIP_NAME || exit
fi

if [ ! -d $FINE_TUNING_DATASET_DIR ]; then
  unzip $FINE_TUNING_DATASET_ZIP_NAME || exit
fi


export PYTHONPATH=.
python3 fine-tune.py --blr 2e-3 --epochs 200 --data_path ./YaTC_datasets/ISCXVPN2016_MFR --nb_classes 7
