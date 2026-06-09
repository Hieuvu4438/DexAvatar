#!/bin/bash
# Download PHOENIX14T Dataset
# ===========================
# PHOENIX14T requires registration at RWTH Aachen University
#
# Steps:
# 1. Visit: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/
# 2. Fill out the registration form (academic use only)
# 3. You will receive a download link via email
# 4. Run this script with the download link
#
# Usage:
#   bash scripts/download_phoenix14t.sh "https://download-link-from-email"
#
# Or download manually and place in:
#   /home/shared_data/sign_language/PHOENIX14T/

set -e

OUTPUT_DIR="/home/shared_data/sign_language/PHOENIX14T"
mkdir -p "$OUTPUT_DIR"

if [ -z "$1" ]; then
    echo "============================================"
    echo "PHOENIX14T Download Instructions"
    echo "============================================"
    echo ""
    echo "PHOENIX14T requires registration. Steps:"
    echo ""
    echo "1. Visit: https://www-i6.informatik.rwth-aachen.de/~koller/RWTH-PHOENIX-2014-T/"
    echo "2. Fill out the registration form"
    echo "3. Wait for email with download link"
    echo "4. Run: bash scripts/download_phoenix14t.sh <download-link>"
    echo ""
    echo "Or download manually and place files in:"
    echo "  $OUTPUT_DIR/"
    echo ""
    echo "Expected structure:"
    echo "  $OUTPUT_DIR/"
    echo "  ├── features/"
    echo "  │   ├── phoenix14t.train"
    echo "  │   ├── phoenix14t.dev"
    echo "  │   └── phoenix14t.test"
    echo "  └── annotations/"
    echo "      └── manual_PHOENIX14T.Corpus.*"
    exit 1
fi

DOWNLOAD_LINK="$1"

echo "Downloading PHOENIX14T to: $OUTPUT_DIR"
echo "This may take a while (~10GB download)..."
echo ""

# Download using wget
wget -c "$DOWNLOAD_LINK" -O "$OUTPUT_DIR/phoenix14t.tar.gz" || \
wget -c "$DOWNLOAD_LINK" -O "$OUTPUT_DIR/phoenix14t.zip"

# Extract
cd "$OUTPUT_DIR"
if [ -f "phoenix14t.tar.gz" ]; then
    echo "Extracting tar.gz..."
    tar -xzf phoenix14t.tar.gz
    rm phoenix14t.tar.gz
elif [ -f "phoenix14t.zip" ]; then
    echo "Extracting zip..."
    unzip -o phoenix14t.zip
    rm phoenix14t.zip
fi

echo ""
echo "Download complete!"
echo "Data location: $OUTPUT_DIR"
ls -la "$OUTPUT_DIR"
