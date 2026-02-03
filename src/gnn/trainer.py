import torch
import torch.nn.functional as F
import os
import json
from typing import Dict
from src.gnn.model import SimpleGCN


class GNNTrainer:
    def __init__(self, config: Dict, num_features: int, num_classes: int):
        self.config = config
        self.device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
        
        # Hyperparameters
        hidden_channels = config.get("hidden_channels", 256)
        num_layers = config.get("num_layers", 3)
        dropout = config.get("dropout", 0.5)
        self.lr = config.get("lr", 0.01)
        self.weight_decay = config.get("weight_decay", 0.0) # Default 0.0
        self.epochs = config.get("epochs", 100)
        
        # Initialize Model
        self.model = SimpleGCN(
            in_channels=num_features,
            hidden_channels=hidden_channels,
            out_channels=num_classes,
            num_layers=num_layers,
            dropout=dropout
        ).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def train(self, data, train_mask) -> float:
        """
        Runs one epoch of training using the provided train_mask.
        Please note that data.y should contain both GT and pseudo labels at the masked positions.
        """
        self.model.train()
        # Move data to device if not already there (though usually we move it once)
        # Here we assume data might be large, passing refs. 
        # For full-batch GCN, we need whole graph on GPU.
        if data.x.device != self.device:
            # Remove raw_text if present, as it cannot be moved to GPU
            if hasattr(data, 'raw_text'):
                del data.raw_text
            data = data.to(self.device)
            
        self.optimizer.zero_grad()
        
        out = self.model(data.x, data.edge_index)
        
        # Loss calculation
        # F.cross_entropy expects class indices (LongTensor)
        # data.y usually is [N, 1], so squeeze it to [N]
        target = data.y.squeeze()
        
        loss = F.cross_entropy(out[train_mask], target[train_mask])
        
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data, mask, true_y=None) -> float:
        """
        Evaluates the model on the given mask and returns accuracy.
        Args:
            data: PyG data object
            mask: Boolean mask for evaluation
            true_y: Optional tensor of True Ground Truth labels. 
                   If None, uses data.y (which might contain pseudo-labels).
        """
        self.model.eval()
        if data.x.device != self.device:
            data = data.to(self.device)
            
        out = self.model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        
        # Use provided true labels if available, otherwise data.y
        target = true_y if true_y is not None else data.y
        target = target.to(self.device).squeeze()
        
        correct = (pred[mask] == target[mask]).sum().item()
        acc = correct / mask.sum().item()
        return acc

    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        """
        Returns predictions for all nodes (or you can slice it later).
        Returns: Tensor [num_nodes] of predicted class indices.
        """
        self.model.eval()
        if data.x.device != self.device:
            data = data.to(self.device)
            
        out = self.model(data.x, data.edge_index)
        # Return class indices
        return out.argmax(dim=1)
    
    @torch.no_grad()
    def get_probs(self, data) -> torch.Tensor:
        """
        Returns soft probabilities/logits if needed for debugging or analysis.
        """
        self.model.eval()
        if data.x.device != self.device:
            data = data.to(self.device)
        return self.model(data.x, data.edge_index)

    def save(self, path: str):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path: str):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    @torch.no_grad()
    def run_gnn_inference(self, data, output_dir, logger):
        """
        Run inference and save predictions to json.
        """
        logger.info("Running GNN Inference (Full Graph)...")
        
        preds = self.predict(data)
        preds_np = preds.cpu().numpy()
        
        # Optional: Save Logits/Softmax for confidence
        # logits = self.get_probs(data)
        # probs = F.softmax(logits, dim=1).cpu().numpy()
        
        predictions = {}
        for idx in range(data.num_nodes):
            predictions[int(idx)] = {
                "gnn_predict": int(preds_np[idx]),
                # "gnn_confidence": float(probs[idx].max()) 
            }
            
        save_path = os.path.join(output_dir, "gnn_predictions.json")
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=2)
            
        logger.info(f"GNN Inference completed. Saved to {save_path}")
        return predictions
