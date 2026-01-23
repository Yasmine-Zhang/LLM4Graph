from typing import Dict


def get_dataset(config: Dict) -> object:
    """
    Factory function to get the dataset loader based on configuration.
    
    Args:
        config (dict): The entire configuration dictionary, but mainly uses config['dataset'].
        
    Returns:
        loader: An instance of a data loader (e.g., ArxivDataLoader).
    """
    dataset_conf = config['dataset']
    name = dataset_conf.get('name', '').lower()
    
    if name == 'ogbn-arxiv':
        from src.data_loader.arxiv_loader import ArxivDataLoader
        return ArxivDataLoader(dataset_conf)
    else:
        raise ValueError(f"Unknown dataset name: {name}")
