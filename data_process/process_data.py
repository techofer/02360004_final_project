import os
from tqdm import tqdm
import pandas as pd
from yatc_preproc import MFR_generator
from my_yatc_train_test_split import yatc_split
from traffic_occlusion.occluder import occlude_CTD
from traffic_occlusion.util import get_sni, mirror_directory_structure, get_pcap_files, get_relative_dest_path
import argparse

def create_ctd_dataset(in_path, out_path):
    mirror_directory_structure(in_path, out_path)
    pcap_files = get_pcap_files(in_path)
    for file_path in tqdm(pcap_files, desc="Processing pcap files", total=len(pcap_files)):
        dest_file = get_relative_dest_path(file_path, in_path, out_path)
        sni = get_sni(file_path)
        occlude_CTD(file_path, dest_file, sni)

def create_yatc_dataset(in_path, out_path) -> bool:
    MFR_generator(in_path, out_path)
    yatc_split(out_path)

def create_bert_dataset(in_path, out_path):
    print("BERT not implemented yet")

def create_datasets(args):
    yatc_path = args.dataset_name + "_YATC"
    ctd_path = args.dataset_name + "_CTD"   
    yatc_ctd_path = args.dataset_name + "_YATC_CTD"
    bert_path = args.dataset_name + "_BERT"
    bert_ctd_path = args.dataset_name + "_BERT_CTD"

    print(f"Creating clean YATC dataset at {yatc_path}")
    create_yatc_dataset(args.src_path, yatc_path)

    print(f"Creating clean BERT dataset at {bert_path}")
    create_bert_dataset(args.src_path, bert_path)

    print(f"Creating raw CTD dataset at {ctd_path}")
    create_ctd_dataset(args.src_path, ctd_path)

    print(f"Creating CTD YATC dataset at {yatc_ctd_path}")
    create_yatc_dataset(ctd_path, yatc_ctd_path)

    print(f"Creating CTD BERT dataset at {bert_ctd_path}")
    create_bert_dataset(ctd_path, bert_ctd_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", help="Path to the flows pcap directory")
    parser.add_argument("--dataset_name", help="Name of the dataset")
    args = parser.parse_args()
    create_datasets(args)

    