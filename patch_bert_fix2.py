import nbformat
from pathlib import Path

path = Path('notebooks/04_modeling.ipynb')
nb = nbformat.read(path, as_version=4)
changed = False
for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    if 'print("\\n📊 Résultats BERT après fine-tuning:")plt.plot(train_losses)' in cell.source:
        cell.source = cell.source.replace(
            'print("\\n📊 Résultats BERT après fine-tuning:")plt.plot(train_losses)',
            'print("\\n📊 Résultats BERT après fine-tuning:")\nplt.plot(train_losses)'
        )
        changed = True
        print('Fixed print/plot newline in cell', i)
    if 'print(classification_report(y_test, bert_pred))plot_enhanced_confusion_matrix(y_test, bert_pred, \'BERT\')' in cell.source:
        cell.source = cell.source.replace(
            'print(classification_report(y_test, bert_pred))plot_enhanced_confusion_matrix(y_test, bert_pred, \'BERT\')',
            'print(classification_report(y_test, bert_pred))\nplot_enhanced_confusion_matrix(y_test, bert_pred, \'BERT\')'
        )
        changed = True
        print('Fixed classification_report/plot newline in cell', i)

if changed:
    nbformat.write(nb, path)
    print('Notebook updated')
else:
    print('No changes made')
