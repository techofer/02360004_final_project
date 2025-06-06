import os
from tqdm import tqdm
import pandas as pd
from scapy.all import rdpcap
from yatc_preproc import MFR_generator
from my_yatc_train_test_split import yatc_split
from traffic_occlusion.occluder import occlude_CTD, occlude_D1, occlude_D2
from traffic_occlusion.util import get_sni, mirror_directory_structure, get_pcap_files, get_relative_dest_path, save_packets
import argparse
from collections import defaultdict


def get_pcap_to_domain(source_dir):
    """
    Recursively retrieve all .pcap files from the source directory.
    """
    domain_to_pcap = defaultdict(list)  
    for root, dirs, files in os.walk(source_dir):
        domain = os.path.basename(root)
        for file in files:
            if file.endswith(".pcap"):
                domain_to_pcap[domain].append(os.path.join(root, file))
    return domain_to_pcap

def create_ctd_dataset(in_path, out_path):
    mirror_directory_structure(in_path, out_path)
    pcap_files = get_pcap_files(in_path)
    for file_path in tqdm(pcap_files, desc="Processing pcap files", total=len(pcap_files)):
        dest_file = get_relative_dest_path(file_path, in_path, out_path)
        sni = get_sni(file_path)
        packets = rdpcap(file_path)
        packets = occlude_CTD(packets, sni)
        save_packets(packets, dest_file)

def create_d12_dataset(in_path, out_path):
    mirror_directory_structure(in_path, out_path)
    domain_to_pcap = get_pcap_to_domain(in_path)
    print(f"Found {len(domain_to_pcap)} domains")

    for domain in tqdm(domain_to_pcap, desc="Processing domain files", total=len(domain_to_pcap)):
        for idx, file_path in enumerate(domain_to_pcap[domain]):
            rel_path = os.path.relpath(file_path, in_path)
            base = os.path.dirname(rel_path)
            new_file = os.path.join(base, f"{idx+1}_occluded.pcap")
            dest_file = os.path.join(out_path, new_file)
            packets = rdpcap(file_path)
            sni = get_sni(file_path)
            packets = occlude_D1(packets)
            if sni:
                packets = occlude_D2(packets, sni)
            save_packets(packets, dest_file)



def create_yatc_dataset(in_path, out_path) -> bool:
    MFR_generator(in_path, out_path)
    yatc_split(out_path)

def create_bert_dataset(in_path, out_path):
    print("BERT not implemented yet")

def create_dataset(args):
    src_path = args.src_path
    print(f"Creating dataset {args.dataset_name} with model {args.model} and occlusion {args.occlusion}")
    dst_path = args.dataset_name + "_" + args.model + "_" + args.occlusion

    if args.occlusion not in src_path:
        print(f"Creating dataset {args.dataset_name} with occlusion {args.occlusion}")
        occ_path = args.dataset_name + "_" + args.occlusion
        if args.occlusion == "CTD":
            create_ctd_dataset(src_path, occ_path)
        elif args.occlusion == "D12":
            create_d12_dataset(src_path, occ_path)
        else:
            raise ValueError(f"Occlusion {args.occlusion} not supported")
        
        src_path = occ_path

    if args.model == "YATC":
        create_yatc_dataset(src_path, dst_path)
    elif args.model == "BERT":
        create_bert_dataset(src_path, dst_path)
    else:
        raise ValueError(f"Model {args.model} not supported")

    print(f"Dataset created in {dst_path}")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", help="Path to the flows pcap directory")
    parser.add_argument("--dataset_name", type=str ,help="Name of the dataset")
    parser.add_argument("--model", choices=["YATC", "BERT"], help="Type of model dataset to create")
    parser.add_argument("--occlusion", choices=["CTD", "D12"], help="Type of occlusion to apply")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_ratio", type=float, default=0.8)

    args = parser.parse_args()
    create_dataset(args)

    