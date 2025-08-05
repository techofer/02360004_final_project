#!/bin/bash

# Configuration
INTERFACE="en0"  # Change to your network interface (check with `ip a`)
DELAY=2           # Time to wait for traffic to complete
CAPTURE_COUNT=335
PCAP_DIR="./blh/mix"

# Ciphers to test
ciphers=(
  "TLS_CHACHA20_POLY1305_SHA256"
  "TLS_AES_256_GCM_SHA384"
  "TLS_AES_128_GCM_SHA256"
)

# Domains to test
domains=(
  adblockplus.org
  doubleclick.net
  google.com
  hsadspixel.net
  naver.com
  steamstatic.com
  web.de
  arin.net
  flipboard.com
  googleapis.com
  hubspot.com
  onetrust.com
  trustarc.com
  wistia.com
  cloudfront.net
  garmin.com
  googletagmanager.com
  joinhoney.com
  qualified.com
  twitter.com
  xnxx-cdn.com
  coinbase.com
  getpocket.com
  gstatic.com
  like-video.com
  robinhood.com
  typekit.net
  yahoo.co.jp
  cookielaw.org
  gmx.net
  hotjar.com
  logitech.com
  segment.com
  ubuntu.com
  yimg.jp
  ctfassets.net
  google-analytics.com
  hs-scripts.com
  mozilla.net
  sharethis.com
  waze.com
)

# Ensure base pcap output path exists
mkdir -p "$PCAP_DIR"

for domain in "${domains[@]}"; do
  domain_dir="${PCAP_DIR}/${domain}"
  mkdir -p "$domain_dir"

  for cipher in "${ciphers[@]}"; do
    echo "[*] Collecting 335 captures for $domain with $cipher..."

    for i in $(seq -f "%04g" 1 "$CAPTURE_COUNT"); do
      filename="${domain}_${cipher}_${i}.pcap"
      filepath="${domain_dir}/${filename}"

      echo "[+] ($i/335) Capturing to $filepath"

      # Start tcpdump in background
      sudo tcpdump -i "$INTERFACE" host "$domain" and port 443 -w "$filepath" > /dev/null 2>&1 &
      TCPDUMP_PID=$!

      sleep "$DELAY"

      # Trigger HTTPS traffic
      curl --connect-timeout 5 --max-time 15 --ciphers "$cipher" -s "https://$domain" > /dev/null 2>&1

      sleep "$DELAY"

      # Kill tcpdump
      sudo kill "$TCPDUMP_PID"
      wait "$TCPDUMP_PID" 2>/dev/null

      # If pcap is empty, remove it
      if [ ! -s "$filepath" ]; then
        echo "[-] Warning: Empty pcap. Removing $filepath"
        rm -f "$filepath"
      fi
    done
  done
done

echo "[✓] All captures complete. Stored in $PCAP_DIR"
