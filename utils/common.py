"""
Common utilities shared across all modules.
Provides manifest writing, logging, config helpers.
"""

import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import subprocess


def setup_logging(level=logging.INFO) -> logging.Logger:
    """Setup consistent logging across modules."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def save_manifest(
    output_dir: Path,
    module_name: str,
    config: Dict[str, Any],
    stats: Optional[Dict[str, Any]] = None
) -> None:
    """
    Save manifest.json with module metadata.
    
    Args:
        output_dir: Output directory path
        module_name: Name of the module (e.g., "00_check_data")
        config: Configuration dictionary
        stats: Optional statistics to include
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    manifest = {
        'module': module_name,
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'stats': stats or {}
    }
    
    # Add git info if available
    try:
        git_hash = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            stderr=subprocess.DEVNULL
        ).decode('utf-8').strip()
        manifest['git_commit'] = git_hash
    except:
        pass
    
    manifest_path = output_dir / 'manifest.json'
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def load_yaml_config(yaml_path: str) -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)


def load_json_data(json_path: str) -> Any:
    """Load JSON data file."""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json_data(data: Any, json_path: str, indent: int = 2) -> None:
    """Save data to JSON file."""
    Path(json_path).parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def print_section(title: str, char: str = '=') -> None:
    """Print formatted section header."""
    print(f"\n{char * 60}")
    print(f"  {title}")
    print(f"{char * 60}\n")
