import nbformat as nbf

nb = nbf.v4.new_notebook()

# Markdown cell
text_intro = """# Advanced Graph ML + NLP (HeteroGNN)
## Relationships: Patient - Drug - Side Effect

This notebook implements an advanced Graph Neural Network (GNN) that models medical reports as a **Heterogeneous Graph**.

### Key Features:
1. **NLP Integration**: We use **Sentence-BERT** (`all-MiniLM-L6-v2`) to create semantic embeddings for Drug names and Side Effect descriptions. This allows the model to "understand" medical terminology similarity.
2. **Heterogeneous Graph**: Nodes are not just patients; we have 3 distinct types:
   - **Patients**: Initialized with demographic/clinical features.
   - **Drugs**: Initialized with semantic embeddings.
   - **Effects**: Initialized with semantic embeddings.
3. **Advanced Message Passing**: Using `HeteroConv` and `GATConv` (Graph Attention Networks) to learn how drugs and symptoms interact to predict patient risk levels."""

nb.cells.append(nbf.v4.new_markdown_cell(text_intro))

# Code cell - imports
code_imports = """import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, GATConv, Linear
import torch_geometric.transforms as T
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")"""

nb.cells.append(nbf.v4.new_code_cell(code_imports))

# Code cell - data loading
code_data = """print("Loading data...")
df = pd.read_csv('../data/processed/dataset_merged_final.csv')

# Preprocessing for graph
df = df.dropna(subset=['all_drugs', 'reaction_meddra', 'target_binary'])
df = df.reset_index(drop=True)
print(f"Loaded {len(df)} reports for graph construction.")"""

nb.cells.append(nbf.v4.new_code_cell(code_data))

# Code cell - NLP embeddings
code_nlp = """print("Generating NLP embeddings (Sentence-BERT)...")
model_nlp = SentenceTransformer('all-MiniLM-L6-v2')

# Process Drugs
all_drugs_list = df['all_drugs'].str.split('|').explode().str.strip().unique()
drug_to_idx = {name: i for i, name in enumerate(all_drugs_list)}
drug_embeddings = model_nlp.encode(all_drugs_list, convert_to_tensor=True)

# Process Effects
all_effects_list = df['reaction_meddra'].str.split(';').explode().str.strip().unique()
effect_to_idx = {name: i for i, name in enumerate(all_effects_list)}
effect_embeddings = model_nlp.encode(all_effects_list, convert_to_tensor=True)

print(f"Nodes: {len(df)} Patients, {len(all_drugs_list)} Drugs, {len(all_effects_list)} Effects")"""

nb.cells.append(nbf.v4.new_code_cell(code_nlp))

# Code cell - graph construction
code_graph = """print("Constructing Heterogeneous Graph...")
data = HeteroData()

# Patient Features
patient_features = ['age', 'weight_kg', 'n_allergies', 'n_chronic_diseases', 'score_risque_interaction']
df[patient_features] = df[patient_features].fillna(df[patient_features].median())
scaler = StandardScaler()
X_patient = scaler.fit_transform(df[patient_features])

data['patient'].x = torch.tensor(X_patient, dtype=torch.float)
data['patient'].y = torch.tensor(df['target_binary'].values, dtype=torch.long)

# Drug and Effect node features (from NLP)
data['drug'].x = drug_embeddings
data['effect'].x = effect_embeddings

# Edges: Patient -> Takes -> Drug
p_idx, d_idx = [], []
for i, row in df.iterrows():
    drugs = str(row['all_drugs']).split('|')
    for d in drugs:
        d = d.strip()
        if d in drug_to_idx:
            p_idx.append(i); d_idx.append(drug_to_idx[d])

data['patient', 'takes', 'drug'].edge_index = torch.tensor([p_idx, d_idx], dtype=torch.long)

# Edges: Patient -> Experiences -> Effect
p_idx_e, e_idx = [], []
for i, row in df.iterrows():
    effects = str(row['reaction_meddra']).split(';')
    for e in effects:
        e = e.strip()
        if e in effect_to_idx:
            p_idx_e.append(i); e_idx.append(effect_to_idx[e])

data['patient', 'experiences', 'effect'].edge_index = torch.tensor([p_idx_e, e_idx], dtype=torch.long)

# Convert to undirected for bi-directional message passing
data = T.ToUndirected()(data)
print(data)"""

nb.cells.append(nbf.v4.new_code_cell(code_graph))

# Code cell - model
code_model = """class HeteroGNN(torch.nn.Module):
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

model = HeteroGNN(hidden_channels=64, out_channels=2, num_layers=2).to(device)
data = data.to(device)
print(model)"""

nb.cells.append(nbf.v4.new_code_cell(code_model))

# Code cell - training
code_train = """optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0]).to(device))

def train():
    model.train()
    optimizer.zero_grad()
    out = model(data.x_dict, data.edge_index_dict)
    loss = criterion(out, data['patient'].y)
    loss.backward()
    optimizer.step()
    return float(loss)

print("Training Advanced HeteroGNN...")
losses = []
for epoch in range(1, 151):
    loss = train()
    losses.append(loss)
    if epoch % 20 == 0:
        print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')

plt.plot(losses)
plt.title("Training Loss")
plt.show()"""

nb.cells.append(nbf.v4.new_code_cell(code_train))

# Code cell - evaluation
code_eval = """model.eval()
with torch.no_grad():
    out = model(data.x_dict, data.edge_index_dict)
    preds = out.argmax(dim=-1).cpu().numpy()
    labels = data['patient'].y.cpu().numpy()
    probs = F.softmax(out, dim=-1)[:, 1].cpu().numpy()

print("\\n--- HeteroGNN Performance ---")
print(classification_report(labels, preds))
print(f"ROC-AUC: {roc_auc_score(labels, probs):.3f}")

# Plot Confusion Matrix
cm = confusion_matrix(labels, preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - HeteroGNN")
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()"""

nb.cells.append(nbf.v4.new_code_cell(code_eval))

# Save notebook
with open('notebooks/06_advanced_graph_nlp_gnn.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Notebook created successfully!")
