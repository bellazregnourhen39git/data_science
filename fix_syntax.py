import json

# Load the notebook
with open('notebooks/04_modeling.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Find the problematic cell (the one with BERT training)
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'bert_pred = np.array(bert_pred)plt.ylabel(\'Loss\')' in ''.join(cell['source']):
        print("Found the problematic cell")
        # Fix the source
        new_source = []
        for line in cell['source']:
            if 'bert_pred = np.array(bert_pred)plt.ylabel(\'Loss\')' in line:
                new_source.append('    bert_pred = np.array(bert_pred)\n')
                new_source.append('    plt.ylabel(\'Loss\')\n')
            else:
                new_source.append(line)
        cell['source'] = new_source
        break

# Save the notebook
with open('notebooks/04_modeling.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print("Fixed the syntax error")