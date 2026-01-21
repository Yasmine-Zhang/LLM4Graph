import os
import yaml
import shutil


def load_config(path):
    """Load YAML configuration file."""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def setup_output_dir(config, exp_name, config_path):
    """
    Create output directory and backup config file.
    Returns the path to the output directory.
    """
    output_dir = os.path.join("output", exp_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy config
    shutil.copy(config_path, os.path.join(output_dir, "config.yaml"))
    
    return output_dir

def parse_node_text(raw_text):
    """
    Parses 'Title: ...\nAbstract: ...' into title and abstract strings.
    """
    try:
        parts = raw_text.split('\nAbstract: ', 1)
        title = parts[0].replace('Title: ', '').strip()
        abstract = parts[1].strip() if len(parts) > 1 else ""
        return title, abstract
    except Exception as e:
        return "", ""
