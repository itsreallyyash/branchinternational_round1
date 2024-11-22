import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.neighbors import BallTree
from sklearn.model_selection import train_test_split
from joblib import dump
import matplotlib.pyplot as plt
import seaborn as sns  # For confusion matrix visualization
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load Data
def load_data():
    file_path = 'csv/engineered_data.csv'
    df = pd.read_csv(file_path)

    # Feature Engineering for Temporal Data
    df['hour_sin'] = np.sin(2 * np.pi * df['application_hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['application_hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['application_dayofweek'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['application_dayofweek'] / 7)

    # Features and Labels
    features = df[['age', 'cash_incoming_30days', 'cash_incoming_per_day', 'distance_traveled',
                  'mean_distance', 'max_distance', 'latitude', 'longitude', 'altitude',
                  'hour_sin', 'hour_cos', 'day_sin', 'day_cos']]
    labels = df['loan_outcome_encoded']

    # Check for NaN values
    if features.isnull().any().any() or labels.isnull().any():
        raise ValueError("Features or labels contain NaN values. Please handle missing data before training.")

    # Ensure labels are binary (0 and 1)
    unique_labels = labels.unique()
    if not set(unique_labels).issubset({0, 1}):
        raise ValueError(f"Labels should be binary encoded as 0 and 1. Found labels: {unique_labels}")

    # Normalize Features
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)

    # Convert to Tensors
    node_features = torch.tensor(features_normalized, dtype=torch.float)
    labels = torch.tensor(labels.values, dtype=torch.long)

    return df, node_features, labels, scaler

# Spatial-Temporal Edge Creation
def create_edges_optimized(features, df, temporal_threshold=1, spatial_threshold=0.1):
    logging.info(f"Starting edge creation with {len(features)} nodes.")

    tree = BallTree(features, metric='euclidean')  # Use all features
    spatial_neighbors = tree.query_radius(features, r=spatial_threshold)

    temporal_features = df['application_hour'].values
    edges = []
    for i, neighbors in enumerate(spatial_neighbors):
        for j in neighbors:
            if i != j and abs(temporal_features[i] - temporal_features[j]) <= temporal_threshold:
                edges.append((i, j))

    if len(edges) == 0:  # Handle case where no edges are created
        logging.warning("No edges were created. Adding self-loops.")
        edges = [(i, i) for i in range(len(features))]  # Add self-loops

    edge_index = torch.tensor(edges, dtype=torch.long).T
    logging.info(f"Edge creation completed. Number of edges: {edge_index.size(1)}")
    return edge_index

# Define Model
class STGCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(STGCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.fc = nn.Linear(out_channels, 2)  # Binary Classification

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.fc(x)
        return x

# Train and Evaluate Model
def train_model(node_features, labels, edge_index, scaler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")

    # Move data to device
    node_features = node_features.to(device)
    labels = labels.to(device)
    edge_index = edge_index.to(device)

    # Create a single Data object with all nodes and edges
    data = Data(x=node_features, edge_index=edge_index, y=labels)

    # Create Train/Test Masks
    train_idx, test_idx = train_test_split(
        np.arange(data.num_nodes), test_size=0.3, stratify=labels.cpu().numpy(), random_state=42
    )
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool, device=device)
    train_mask[train_idx] = True  # Correctly set training indices
    test_mask[test_idx] = True    # Correctly set testing indices
    data.train_mask = train_mask
    data.test_mask = test_mask

    logging.info(f"Number of training samples: {train_mask.sum().item()}")
    logging.info(f"Number of testing samples: {test_mask.sum().item()}")

    # Add Self-Loops
    data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.num_nodes)

    # Initialize Model
    model = STGCN(in_channels=node_features.shape[1], hidden_channels=32, out_channels=16).to(device)
    model.fc = nn.Linear(16, 2).to(device)  # Ensure the final layer has 2 outputs

    # Use DataParallel for multi-GPU if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logging.info(f"Using {torch.cuda.device_count()} GPUs for training.")

    # Optimizer and Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    # Optionally, use mixed precision for faster training
    scaler_amp = torch.cuda.amp.GradScaler()

    # Training Loop
    epochs = 50
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            out = model(data.x, data.edge_index)
            loss = criterion(out[data.train_mask], data.y[data.train_mask])
        scaler_amp.scale(loss).backward()
        scaler_amp.step(optimizer)
        scaler_amp.update()
        logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

    # Save Model and Scaler
    torch.save(model.state_dict(), "stgcn_model.pth")
    dump(scaler, "scaler.joblib")
    logging.info("Model and scaler saved.")

    # Evaluation
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        probs = torch.softmax(out, dim=1)
        y_proba = probs[:, 1][data.test_mask].cpu().numpy()
        y_pred = out.argmax(dim=1)[data.test_mask].cpu().numpy()
        y_true = data.y[data.test_mask].cpu().numpy()

    # Ensure that y_true and y_pred are not empty
    if len(y_true) == 0:
        logging.error("No samples in test set. Check your train/test split.")
        return

    # Metrics
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=["Not Defaulted", "Defaulted"]))
    roc_auc = roc_auc_score(y_true, y_proba)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_true, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="coolwarm",
                xticklabels=["Not Defaulted", "Defaulted"], yticklabels=["Not Defaulted", "Defaulted"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

if __name__ == "__main__":
    try:
        # Load Data and Features
        df, node_features, labels, scaler = load_data()

        # Create Edges
        edge_index = create_edges_optimized(node_features.numpy(), df)

        # Verify Edge Indices
        data_object = Data(x=node_features, edge_index=edge_index, y=labels)
        assert edge_index.max() < data_object.num_nodes, "Edge index has out-of-bounds node indices."
        logging.info("Edge indices are valid.")

        # Train and Evaluate
        train_model(node_features, labels, edge_index, scaler)

    except Exception as e:
        logging.error(f"An error occurred: {e}")
