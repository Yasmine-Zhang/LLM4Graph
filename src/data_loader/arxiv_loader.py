import os
import pandas as pd
import torch
from ogb.nodeproppred import PygNodePropPredDataset

# MONKEY PATCH: Fix for PyTorch 2.4+ safe load issue with OGB/PyG
# OGB uses torch.load() which defaults to weights_only=True in newer PyTorch versions,
# breaking the loading of complex PyG data objects.
_original_load = torch.load
def _unsafe_load(*args, **kwargs):
    # Only inject weights_only=False if it's not already specified
    if 'weights_only' not in kwargs:
        kwargs['weights_only'] = False
    return _original_load(*args, **kwargs)
torch.load = _unsafe_load


class ArxivDataLoader:
    def __init__(self, root='./dataset'):
        self.root = root
        print("Loading OGB graph structure...")
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        self.data = self.dataset[0]
        self.split_idx = self.dataset.get_idx_split()
        self.label_map = self.get_label_mapping()
        self.text = self.get_raw_text()

    def get_formatted_message(self, node_idx: int) -> str:
        """
        Generates the message content for a specific node to be sent to the LLM.
        Directly uses the raw text (Title + Abstract) for now.
        
        Args:
            node_idx (int): The index of the node in the graph.
            
        Returns:
            str: The formatted message string.
        """
        # Ensure raw_text is available and index is within bounds
        if not self.text or node_idx >= len(self.text):
            return ""
        return self.text[node_idx]

    def get_system_prompt(self) -> str:
        """
        Generates the system prompt that defines the task and output format for the LLM.
        Includes the list of valid categories from the label mapping.
        
        Returns:
            str: The system prompt string.
        """
        label_list_str = "\n".join([f"{k}: {v}" for k, v in self.label_map.items()]) if self.label_map else "No labels available."
        
        prompt = (
            "You are an expert in computer science research paper classification. "
            "Your task is to predict the category of a paper based on its title and abstract.\n\n"
            "The available categories are:\n"
            f"{label_list_str}\n\n"
            "Please analyze the provided paper and predict the most likely category. "
            "Output your answer in JSON format with two keys: 'category_id' (int) and 'confidence' (float between 0 and 1)."
        )
        return prompt

    def get_label_mapping(self):
        """
        Loads the mapping from label index to arxiv category string.
        
        Returns:
            dict: A dictionary where keys are label indices (int) and values are category names (str).
                  e.g., {0: 'arxiv cs.AI', 1: 'arxiv cs.CL', ...}
        """
        mapping_path = os.path.join(self.root, 'ogbn_arxiv', 'mapping', 'labelidx2arxivcategeory.csv.gz')
        if not os.path.exists(mapping_path):
             return None
        
        df = pd.read_csv(mapping_path)
        return dict(zip(df['label idx'], df['arxiv category']))

    def get_inv_label_map(self) -> dict:
        """
        Returns a mapping from standardized category string to label index.
        Standardizes 'arxiv cs ai' -> 'cs.AI' for consistency with LLM outputs.
        Example: 'arxiv cs na' -> 'CS.NA'
        """
        if not self.label_map:
            return {}
        
        inv_map = {}
        for idx, cat_str in self.label_map.items():
            # Logic: 'arxiv cs na' -> 'CS.NA'
            # 1. Upper case -> 'ARXIV CS NA'
            # 2. Remove 'ARXIV ' -> 'CS NA'
            # 3. Replace space with dot -> 'CS.NA'
            clean_cat = cat_str.upper().replace('ARXIV ', '').replace(' ', '.')
            inv_map[clean_cat] = idx
        return inv_map

    def get_raw_text(self):
        """
        Loads the raw text (title and abstract) for the arxiv papers.
        Downloads and extracts the data if it doesn't exist.
        
        Returns:
            List[str]: A list of strings, where each string is the formatted text for a node.
                       order corresponds to the node index.
        """
        raw_dir = os.path.join(self.root, 'ogbn_arxiv', 'raw')
        mapping_path = os.path.join(self.root, 'ogbn_arxiv', 'mapping', 'nodeidx2paperid.csv.gz')
        text_path = os.path.join(raw_dir, 'titleabs.tsv')

        if not os.path.exists(text_path):
            print(f"Raw text not found at {text_path}. Downloading...")
            from ogb.utils.url import download_url
            import tarfile
            
            # SNAP URL
            url = "http://snap.stanford.edu/ogb/data/misc/ogbn_arxiv/titleabs.tsv.gz"
            file_path = download_url(url, raw_dir)
            
            # Extract
            print("Extracting tarball...")
            with tarfile.open(file_path, "r:gz") as tar:
                tar.extractall(path=raw_dir)
            print("Download and extraction complete.")

        print("Reading raw text and aligning...")
        
        nodeidx2paperid = pd.read_csv(mapping_path)
        raw_text = pd.read_csv(text_path, sep='\t', header=None, names=['paper id', 'title', 'abs'], quoting=3)
        nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype(int)
        raw_text['paper id'] = raw_text['paper id'].astype(int)
        df = pd.merge(nodeidx2paperid, raw_text, on='paper id', how='left')
        df['title'] = df['title'].fillna('')
        df['abs'] = df['abs'].fillna('')
        text_list = []
        for ti, ab in zip(df['title'], df['abs']):
            text_list.append(f"Title: {ti}\nAbstract: {ab}")
            
        return text_list

    def get_data(self):
        """
        Constructs and returns the PyG Data object for the graph.
        Includes graph structure, features, labels, masks, and raw text.
        
        Returns:
            torch_geometric.data.Data: The graph data object.
        """
        from torch_geometric.data import Data
        data = Data(
            x=self.data.x, 
            edge_index=self.data.edge_index,
            y=self.data.y,
            num_nodes=self.data.num_nodes
        )
        data.raw_text = self.text
        data.train_mask = self._idx_to_mask(self.dataset.num_nodes, self.split_idx['train'])
        data.val_mask = self._idx_to_mask(self.dataset.num_nodes, self.split_idx['valid'])
        data.test_mask = self._idx_to_mask(self.dataset.num_nodes, self.split_idx['test'])
        data.num_classes = self.dataset.num_classes
        
        return data

    def _idx_to_mask(self, num_nodes, idx):
        """
        Helper method to convert index tensor to boolean mask.
        
        Args:
            num_nodes (int): Total number of nodes.
            idx (torch.Tensor): Tensor of indices to be set to True.
            
        Returns:
            torch.Tensor: Boolean mask of shape (num_nodes,).
        """
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        return mask

