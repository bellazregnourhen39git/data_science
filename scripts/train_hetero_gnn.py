import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os

# --- 1. Load Data ---
print("Loading data...")
df = pd.read_csv('data/processed/dataset_merged_final.csv')

# Drop rows with missing crucial info for the graph
df = df.dropna(subset=['all_drugs', 'reaction_meddra', 'target_binary'])

# --- 2. NLP Embeddings for Drugs and Effects ---
print("Generating NLP embeddings...")
model_nlp = SentenceTransformer('all-MiniLM-L6-v2')

# Process Drugs
all_drugs_list = df['all_drugs'].str.split('|').explode().str.strip().unique()
drug_to_idx = {name: i for i, name in enumerate(all_drugs_list)}
drug_embeddings = model_nlp.encode(all_drugs_list, convert_to_tensor=True)

# Process Effects
all_effects_list = df['reaction_meddra'].str.split(';').explode().str.strip().unique()
effect_to_idx = {name: i for i, name in enumerate(all_effects_list)}
effect_embeddings = model_nlp.encode(all_effects_list, convert_to_tensor=True)

# --- 3. Build Heterogeneous Graph ---
print("Building graph...")
data = HeteroData()

# Patient nodes
# Features: age, sex, weight, n_allergies, etc.
patient_features = ['age', 'weight_kg', 'n_allergies', 'n_chronic_diseases', 'score_risque_interaction']
df[patient_features] = df[patient_features].fillna(df[patient_features].median())
scaler = StandardScaler()
X_patient = scaler.fit_transform(df[patient_features])
data['patient'].x = torch.tensor(X_patient, dtype=torch.float)
data['patient'].y = torch.tensor(df['target_binary'].values, dtype=torch.long)

# Drug nodes
data['drug'].x = drug_embeddings

# Effect nodes
data['effect'].x = effect_embeddings

# Edges: Patient -> Takes -> Drug
p_idx = []
d_idx = []
for i, row in df.iterrows():
    drugs = str(row['all_drugs']).split('|')
    for d in drugs:
        d = d.strip()
        if d in drug_to_idx:
            p_idx.append(i)
            d_idx.append(drug_to_idx[d])

data['patient', 'takes', 'drug'].edge_index = torch.tensor([p_idx, d_idx], dtype=torch.long)

# Edges: Patient -> Experiences -> Effect
p_idx_e = []
e_idx = []
for i, row in df.iterrows():
    effects = str(row['reaction_meddra']).split(';')
    for e in effects:
        e = e.strip()
        if e in effect_to_idx:
            p_idx_e.append(i)
            e_idx.append(effect_to_idx[e])

data['patient', 'experiences', 'effect'].edge_index = torch.tensor([p_idx_e, e_idx], dtype=torch.long)

# Add reverse edges for message passing
import torch_geometric.transforms as T
data = T.ToUndirected()(data)

# --- 4. Define HeteroGNN Model ---
class HeteroGNN(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, num_layers):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HeteroConv({
                ('patient', 'takes', 'drug'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('drug', 'rev_takes', 'patient'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('patient', 'experiences', 'effect'): GATConv((-1, -1), hidden_channels, add_self_loops=False),
                ('effect', 'rev_experiences', 'patient'): GATConv((-1, -1), hidden_channels, hidden_channels, add_self_loops=False),
            }, aggr='mean')
            self.convs.append(conv)
            
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}
        return self.lin(x_dict['patient'])

# --- 5. Training ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = HeteroGNN(hidden_channels=64, out_channels=2, num_layers=2).to(device)
data = data.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 5.0]).to(device)) # Handle imbalance

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out, data['patient'].y)
    loss.backward()
    optimizer.step()
    return float(loss)

print("Starting training...")
for epoch in range(1, 101):
    loss = train()
    if epoch % 10 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

# --- 6. Evaluation ---
model.eval()
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)
    preds = out.argmax(dim=-1).cpu().numpy()
    labels = data['patient'].y.cpu().numpy()
    probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

print("\nModel Evaluation:")
print(classification_report(labels, preds))
print(f"ROC-AUC: {roc_auc_score(labels, probs):.3f}")

# Save model
torch.save(model.state_dict(), 'best_hetero_gnn.pth')
print("Model saved to best_hetero_gnn.pth")
