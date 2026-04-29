@echo off
cd "c:\Users\rania\OneDrive\Desktop\rania\datascienceproject\data_science"
powershell -Command "(Get-Content 'notebooks\04_modeling.ipynb') -replace 'bert_pred = np.array\(bert_pred\)plt.ylabel\(''Loss''\)', 'bert_pred = np.array(bert_pred)\n    plt.ylabel(''Loss'')' | Set-Content 'notebooks\04_modeling.ipynb'"