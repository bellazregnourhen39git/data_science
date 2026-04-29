import nbformat
from pathlib import Path

path = Path('notebooks/04_modeling.ipynb')
nb = nbformat.read(path, as_version=4)

for i, cell in enumerate(nb.cells):
    if cell.cell_type != 'code':
        continue
    src = cell.source
    if 'from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments' in src:
        cell.source = src.replace(
            'from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments',
            'from transformers import BertTokenizer, BertModel, BertForSequenceClassification'
        )
        print('Patched import cell', i)
    if 'print("🧠 Fine-tuning de BERT pour analyse NLP améliorée...")' in src and 'trainer = Trainer(' in src:
        cell.source = '''# Modèle BERT amélioré avec fine-tuning
print("🧠 Fine-tuning de BERT pour analyse NLP améliorée...")

bert_model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=2,
    hidden_dropout_prob=0.3,
    attention_probs_dropout_prob=0.3
)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def tokenize_for_bert(texts, max_length=128):
    tokenized = []
    for text in texts:
        text = str(text) if pd.notna(text) else "no symptoms reported"
        text = text.lower().strip()
        inputs = tokenizer(
            text,
            return_tensors='pt',
            max_length=max_length,
            padding='max_length',
            truncation=True,
            add_special_tokens=True
        )
        tokenized.append({
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze()
        })
    return tokenized

print("🔄 Tokenization améliorée des textes pour BERT...")
train_tokenized = tokenize_for_bert(df.iloc[X_train.index]['symptoms_text'].values)
test_tokenized = tokenize_for_bert(df.iloc[X_test.index]['symptoms_text'].values)

train_input_ids = torch.stack([item['input_ids'] for item in train_tokenized])
train_attention_mask = torch.stack([item['attention_mask'] for item in train_tokenized])
train_labels_bert = torch.tensor(y_train.values, dtype=torch.long)

test_input_ids = torch.stack([item['input_ids'] for item in test_tokenized])
test_attention_mask = torch.stack([item['attention_mask'] for item in test_tokenized])
test_labels_bert = torch.tensor(y_test.values, dtype=torch.long)

print(f"✅ Données BERT préparées: {len(train_tokenized)} train, {len(test_tokenized)} test")

class BERTDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx],
            'labels': self.labels[idx]
        }

train_dataset = BERTDataset(train_input_ids, train_attention_mask, train_labels_bert)
test_dataset = BERTDataset(test_input_ids, test_attention_mask, test_labels_bert)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

optimizer_bert = optim.AdamW(bert_model.parameters(), lr=2e-5, weight_decay=0.01)
criterion_bert = nn.CrossEntropyLoss()

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
        total += labels.size(0)

    return total_loss / total


def evaluate(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            total_loss += loss.item() * input_ids.size(0)
            total += labels.size(0)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return total_loss / total, np.array(all_preds), np.array(all_labels)

print("🚀 Fine-tuning de BERT...")
epochs = 3
train_losses = []
for epoch in range(1, epochs + 1):
    train_loss = train_epoch(bert_model, train_loader, optimizer_bert, criterion_bert)
    train_losses.append(train_loss)
    print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f}")

val_loss, bert_pred, bert_labels = evaluate(bert_model, test_loader)
print(f"Validation loss: {val_loss:.4f}")

bert_proba = []
bert_model.eval()
with torch.no_grad():
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=1)[:, 1]
        bert_proba.extend(probs.cpu().numpy())

bert_pred = np.array(bert_pred)
bert_proba = np.array(bert_proba)

print("\n📊 Résultats BERT après fine-tuning:")
print(classification_report(y_test, bert_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, bert_proba):.3f}")

plot_enhanced_confusion_matrix(y_test, bert_pred, 'BERT')
plt.plot(train_losses)
plt.title('Évolution de la perte - BERT')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()
'''
        print('Patched BERT Trainer cell', i)
    if 'def train_bert(model, train_input_ids, train_attention_mask, train_labels' in src and 'bert_classifier' in src:
        cell.source = '# Cette cellule est devenue obsolète après la correction du workflow BERT ci-dessus.\n'
        print('Replaced duplicate BERT cell', i)

nbformat.write(nb, path)
print('Done')
