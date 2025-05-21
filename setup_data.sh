#!/bin/bash

sudo apt-get install git tshark -y

pip install pip -U
pip install scapy pyshark tqdm xlrd flowcontainer

if [ ! -d ntc-enigma ]; then
  git clone https://github.com/nime-sha256/ntc-enigma.git || exit
fi 

cd ntc-enigma/traffic_occlusion || exit

PYTHONPATH=.
mkdir -p ../../oracle.com_occluded/a/b/
mkdir -p ../../oracle.com_occluded_MFR/a/b/
python3 main.py ../../oracle.com ../../oracle.com_occluded/a/b/ --option D2

cd ../../YaTC || exit

PYTHONPATH=.
python3 -c 'from data_process import MFR_generator; MFR_generator("../oracle.com_occluded/", "../oracle.com_occluded_MFR/")' || exit

cd ../ET-BERT/data_process || exit
export PYTHONPATH=.

python3 main.py