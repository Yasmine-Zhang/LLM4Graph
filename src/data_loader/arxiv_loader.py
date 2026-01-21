import os.path as osp
import pandas as pd
import torch
import os
from ogb.nodeproppred import PygNodePropPredDataset

class ArxivDataLoader:
    def __init__(self, root='./dataset'):
        self.root = root
        print("Loading OGB graph structure...")
        self.dataset = PygNodePropPredDataset(name='ogbn-arxiv', root=root)
        self.data = self.dataset[0]
        self.split_idx = self.dataset.get_idx_split()
        self.text = self.get_raw_text()

    def get_raw_text(self):
        raw_dir = osp.join(self.root, 'ogbn_arxiv', 'raw')
        mapping_path = osp.join(self.root, 'ogbn_arxiv', 'mapping', 'nodeidx2paperid.csv.gz')
        text_path = osp.join(raw_dir, 'titleabs.tsv')

        if not osp.exists(text_path):
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
        mask = torch.zeros(num_nodes, dtype=torch.bool)
        mask[idx] = True
        return mask

