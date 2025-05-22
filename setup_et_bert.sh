#!/bin/bash

PRETRAINED_MODEL_NAME=pretrained_model.bin
FINE_TUNING_DATASET_ZIP_NAME=cstnet-tls1.3_packet.zip
FINE_TUNING_DATASET_DIR_NAME=cstnet-tls1.3_packet

sudo apt-get install git unzip -y

pip install pip -U
pip install six "gdown>=5.2.0" "numpy<2.0" "torch>=2.7.0" "tqdm>=4.67.1"

if [ ! -f $PRETRAINED_MODEL_NAME ]; then
  gdown https://drive.google.com/uc?id=1r1yE34dU2W8zSqx1FkB8gCWri4DQWVtE -O $PRETRAINED_MODEL_NAME  || exit
fi

if [ ! -f $FINE_TUNING_DATASET_ZIP_NAME ]; then
  gdown https://drive.google.com/uc?id=1rZX1Y5v-4eTUX9S6VNEk3xD-UYv2MoC4 -O $FINE_TUNING_DATASET_ZIP_NAME || exit
fi

if [ ! -d $FINE_TUNING_DATASET_DIR_NAME ]; then
  unzip $FINE_TUNING_DATASET_ZIP_NAME -d $FINE_TUNING_DATASET_DIR_NAME || exit
fi

if [ ! -d ET-BERT ]; then
  git clone https://github.com/linwhitehat/ET-BERT.git || exit
fi

cd ET-BERT || exit
export PYTHONPATH=.
python3 fine-tuning/run_classifier.py \
  --pretrained_model_path ../$PRETRAINED_MODEL_NAME \
  --train_path ../$FINE_TUNING_DATASET_DIR_NAME/packet/train_dataset.tsv \
  --dev_path ../$FINE_TUNING_DATASET_DIR_NAME/packet/valid_dataset.tsv \
  --vocab_path models/encryptd_vocab.txt \
  --epochs_num 10 --batch_size 32 --embedding word_pos_seg \
  --encoder transformer --mask fully_visible \
  --seq_length 128 --learning_rate 2e-5

