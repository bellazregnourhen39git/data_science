import json
import os
import sys

notebook_path = r'c:\Users\rania\OneDrive\Desktop\rania\datascienceproject\data_science\notebooks\04_modeling.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb['cells']:
    source_text = "".join(cell['source']).lower()
    
    # Skip cells purely about BERT or DistilBERT training
    if ("distilbert" in source_text or "bertclassifier" in source_text or "bert_model" in source_text) and "fusion" not in source_text:
        continue
    
    # Clean up source from BERT references
    new_source = []
    for line in cell['source']:
        # Skip lines defining or using BERT proba/pred in fusion contexts
        if "'bert_proba': bert_proba" in line or "'bert_pred': bert_pred" in line:
            continue
        if "'bert_error':" in line.lower():
            continue
        if "'bert'" in line or "bert fine-tuné" in line or "bert fine-tune" in line:
            # If it's a list item, we need to be careful with commas
            if "'bert'" in line.lower() and ("[" in line or "," in line):
                line = line.replace("'BERT',", "").replace("'BERT'", "").replace("'BERT Fine-tuné',", "").replace("'BERT Fine-tuné'", "")
            else:
                continue
        if "bert_proba" in line and "auc_scores" in line:
            continue
        if "bert_pred" in line and "f1_scores" in line:
            continue
        
        new_source.append(line)
    
    cell['source'] = new_source
    
    # Check if cell is now empty or just comments, if so skip it (except if it was fusion)
    clean_text = "".join(new_source).strip()
    if not clean_text and "fusion" not in source_text:
        continue

    # Clear output of modified cells
    if "fusion" in source_text or "comparaison" in source_text:
        cell['outputs'] = []
        cell['execution_count'] = None

    new_cells.append(cell)

# Ensure the last cell added is the test cell
test_cell_id = "test_prediction_cell"
# Remove if already exists to avoid duplicates
new_cells = [c for c in new_cells if c.get('id') != test_cell_id]

test_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": test_cell_id,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 🧪 TEST DE PRÉDICTION FINAL (SANS BERT)\n",
        "print(\"🧪 Test du modèle de fusion sur des cas réels du set de test...\")\n",
        "\n",
        "import random\n",
        "import pandas as pd\n",
        "\n",
        "try:\n",
        "    # Sélectionner 5 exemples aléatoires\n",
        "    n_samples = 5\n",
        "    indices = random.sample(range(len(X_test)), n_samples)\n",
        "\n",
        "    print(f\"\\n📋 Analyse de {n_samples} prédictions :\")\n",
        "    print(\"-\" * 80)\n",
        "\n",
        "    for idx in indices:\n",
        "        orig_idx = X_test.index[idx]\n",
        "        info = df.iloc[orig_idx]\n",
        "        \n",
        "        p_xgb = xgb_proba[idx]\n",
        "        p_rf = rf_proba[idx]\n",
        "        p_gnn = gnn_proba[idx]\n",
        "        \n",
        "        prob = fusion_proba[idx]\n",
        "        pred = fusion_pred[idx]\n",
        "        true = y_test.iloc[idx]\n",
        "        \n",
        "        print(f\"👤 Patient ID: {info.get('patient_id', 'N/A')}\")\n",
        "        print(f\"📝 Symptômes: {str(info.get('symptoms_text', 'N/A'))[:80]}...\")\n",
        "        print(f\"📊 Modèles: XGB={p_xgb:.2f}, RF={p_rf:.2f}, GNN={p_gnn:.2f}\")\n",
        "        print(f\"🎯 Réel: {'Grave' if true == 1 else 'Normal'} | 🔮 Fusion: {'Grave' if pred == 1 else 'Normal'} ({prob*100:.1f}%)\")\n",
        "        print(f\"✅ Résultat: {'CORRECT' if pred == true else 'ERREUR'}\")\n",
        "        print(\"-\" * 80)\n",
        "except Exception as e:\n",
        "    print(f\"❌ Erreur lors du test: {e}\")\n",
        "    print(\"Assurez-vous d'avoir exécuté la cellule de Fusion au préalable.\")\n"
    ]
}

new_cells.append(test_cell)

nb['cells'] = new_cells

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook updated successfully.")
