import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import logging
from graph_attention_network import GraphAttentionNetwork
from train_evaluate import train_model, evaluate_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    dataset_name = 'Cora'
    dataset = Planetoid(root='/tmp/' + dataset_name, name=dataset_name)
    dataset.transform = T.NormalizeFeatures()

    logger.info(f"Dataset: {dataset_name}")
    logger.info(f"Number of Classes: {dataset.num_classes}")
    logger.info(f"Number of Node Features: {dataset.num_node_features}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = dataset[0].to(device)

    model = GraphAttentionNetwork(
        input_features=dataset.num_node_features,
        hidden_units=8,
        num_classes=dataset.num_classes
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=5e-4)

    train_model(model, data, optimizer, epochs=1000)

    accuracy = evaluate_model(model, data)
    logger.info(f'Test Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    main()