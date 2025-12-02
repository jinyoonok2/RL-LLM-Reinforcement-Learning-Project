#!/usr/bin/env python3
"""
Backup outputs folder to zip and push to git.
"""

import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def backup_training_data():
    """Backup outputs and datasets (candidates) to git."""
    
    # Paths to backup
    outputs_dir = Path("outputs")
    datasets_dir = Path("datasets")
    
    # Create zip files
    zip_files = []
    
    # Backup outputs
    if outputs_dir.exists():
        outputs_zip = "outputs.zip"
        if Path(outputs_zip).exists():
            Path(outputs_zip).unlink()
        shutil.make_archive("outputs", 'zip', outputs_dir)
        zip_files.append(outputs_zip)
        print(f"✅ Created {outputs_zip}")
    
    # Backup datasets (candidates)
    if datasets_dir.exists():
        datasets_zip = "datasets.zip"
        if Path(datasets_zip).exists():
            Path(datasets_zip).unlink()
        shutil.make_archive("datasets", 'zip', datasets_dir)
        zip_files.append(datasets_zip)
        print(f"✅ Created {datasets_zip}")
    
    if not zip_files:
        print("❌ No outputs or datasets folders found")
        return
    
    # Git operations
    try:
        # Add zip files
        for zip_file in zip_files:
            subprocess.run(['git', 'add', zip_file], check=True)
        
        # Commit
        commit_msg = f"Backup training data - SFT complete (acc=92.98%, reward=95.46%)"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        
        # Push
        subprocess.run(['git', 'push', 'origin', 'main'], check=True)
        
        print(f"🚀 Pushed {', '.join(zip_files)} to repository")
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Git operation failed: {e}")

if __name__ == "__main__":
    backup_training_data()