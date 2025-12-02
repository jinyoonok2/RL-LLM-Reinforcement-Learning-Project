#!/usr/bin/env python3
"""
Restore backup files by unzipping outputs.zip and datasets.zip back to their original structure.

Usage:
    python restore_backup.py
    python restore_backup.py --force  # Overwrite existing folders
"""

import argparse
import shutil
import zipfile
from pathlib import Path


def restore_from_backup(force: bool = False):
    """Restore outputs and datasets folders from zip files."""
    
    zip_files = [
        ("outputs.zip", "outputs"),
        ("datasets.zip", "datasets")
    ]
    
    restored = []
    
    for zip_file, target_dir in zip_files:
        zip_path = Path(zip_file)
        target_path = Path(target_dir)
        
        if not zip_path.exists():
            print(f"⚠️  {zip_file} not found, skipping")
            continue
            
        # Check if target directory exists
        if target_path.exists():
            if not force:
                print(f"⚠️  {target_dir}/ already exists, use --force to overwrite")
                continue
            else:
                print(f"🗑️  Removing existing {target_dir}/")
                shutil.rmtree(target_path)
        
        # Extract zip file
        print(f"📦 Extracting {zip_file} to {target_dir}/")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(".")
            
            print(f"✅ Restored {target_dir}/ from {zip_file}")
            restored.append(target_dir)
            
        except Exception as e:
            print(f"❌ Failed to extract {zip_file}: {e}")
    
    if restored:
        print(f"🎉 Successfully restored: {', '.join(restored)}")
    else:
        print("📭 No backups were restored")


def main():
    parser = argparse.ArgumentParser(description="Restore outputs and datasets from backup zip files")
    parser.add_argument("--force", action="store_true",
                       help="Overwrite existing folders if they exist")
    args = parser.parse_args()
    
    restore_from_backup(args.force)


if __name__ == "__main__":
    main()