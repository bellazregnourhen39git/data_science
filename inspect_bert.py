import nbformat
from pathlib import Path
p = Path('notebooks/04_modeling.ipynb')
nb = nbformat.read(p, as_version=4)
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    if any(keyword in cell.source for keyword in ['bert_model.to(device)', 'TrainingArguments', 'trainer = Trainer', 'train_bert(', 'bert_classifier', 'plot(train_losses)', 'print(classification_report']):
        print('--- CELL', i, '---')
        print(cell.source)
        print()
