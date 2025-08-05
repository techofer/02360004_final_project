#!/bin/bash

input_root="mix"
output_root="spectrum_tls_only"

# Create output root if it doesn't exist
mkdir -p "$output_root"

# Loop over all pcap files under mix/<domain>/
find "$input_root" -type f -name "*.pcap" | while read -r pcap_path; do
    # Extract domain and filename
    relative_path="${pcap_path#$input_root/}"             # remove 'mix/' prefix
    domain="${relative_path%%/*}"                         # get the first part (domain)
    filename="$(basename "$pcap_path")"                   # extract filename
    filename_no_ext="${filename%.*}"                      # remove .pcap extension

    # Define output path
    output_dir="$output_root/$domain"
    output_file="$output_dir/${filename_no_ext}_tls_only.pcap"

    # Create output directory if needed
    mkdir -p "$output_dir"

    # Get the first TLS stream index
    stream_index=$(tshark -r "$pcap_path" -Y tls -T fields -e tcp.stream | sort -n | uniq | head -n 1)

    # If no TLS stream found, skip
    if [ -z "$stream_index" ]; then
        echo "No TLS stream found in $pcap_path"
        continue
    fi

    # Extract only TLS packets from the first stream
    tshark -r "$pcap_path" -Y "tls && tcp.stream == $stream_index" -w "$output_file"
    echo "Extracted TLS stream $stream_index from $pcap_path → $output_file"
done
