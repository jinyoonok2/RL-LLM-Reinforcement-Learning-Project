#!/bin/bash
# Unzip outputs.zip to restore outputs folder

if [ ! -f "outputs.zip" ]; then
    echo "Error: outputs.zip not found!"
    exit 1
fi

echo "Unzipping outputs.zip..."
unzip -q outputs.zip

if [ $? -eq 0 ]; then
    echo "✅ Successfully unzipped outputs.zip"
    echo "Contents:"
    ls -lh outputs/
else
    echo "❌ Failed to unzip outputs.zip"
    exit 1
fi
