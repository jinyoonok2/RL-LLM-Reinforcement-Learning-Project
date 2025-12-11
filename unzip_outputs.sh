#!/bin/bash
# Unzip outputs.zip to restore outputs folder

if [ ! -f "outputs.zip" ]; then
    echo "Error: outputs.zip not found!"
    exit 1
fi

# Create outputs directory if it doesn't exist
mkdir -p outputs

echo "Unzipping outputs.zip..."
unzip -o outputs.zip -d .

if [ $? -eq 0 ]; then
    echo "✅ Successfully unzipped outputs.zip"
    echo ""
    echo "Contents of outputs/:"
    ls -lh outputs/
    echo ""
    if [ -d "outputs/run_001" ]; then
        echo "run_001 contents:"
        ls -lh outputs/run_001/
    fi
    if [ -d "outputs/run_002" ]; then
        echo "run_002 contents:"
        ls -lh outputs/run_002/
    fi
else
    echo "❌ Failed to unzip outputs.zip"
    exit 1
fi
