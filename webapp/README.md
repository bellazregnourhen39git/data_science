# Medical Dashboard Web App

Ce dossier contient une application web complète pour votre projet de data science médicale.

## Architecture

- `backend/`: API Flask qui sert les prédictions et les métriques.
- `frontend/`: Interface React moderne avec dashboard, formulaire de test et résultats.

## Installation

### Backend

1. Placez-vous dans le dossier `data_science/webapp/backend`.
2. Créez un environnement Python et activez-le.
3. Installez les dépendances :

```bash
pip install -r requirements.txt
```

4. Lancez le backend :

```bash
python app.py
```

### Frontend

1. Placez-vous dans le dossier `data_science/webapp/frontend`.
2. Installez les dépendances :

```bash
npm install
```

3. Lancez l'application React :

```bash
npm start
```

## API

- `GET /api/health` : vérifie que le backend fonctionne.
- `GET /api/metrics` : renvoie les métriques du tableau de bord.
- `POST /api/predict` : envoie les données du patient et récupère la prédiction.

### Exemple de payload pour `/api/predict`

```json
{
  "age": 45,
  "bmi": 24.5,
  "n_drugs": 2,
  "n_symptoms": 3,
  "severity_score": 3,
  "vulnerability_score": 4,
  "symptoms_text": "Douleur, fièvre, fatigue"
}
```

## Intégration de vos modèles

Le backend peut charger des modèles pré-entraînés dans `backend/models/` si vous sauvegardez :

- `xgb_model.joblib`
- `rf_model.joblib`
- `fusion_model.joblib`

Si aucun modèle n'est trouvé, l'API utilise une prédiction heuristique pour permettre un test immédiat.

## Notes

- Le frontend est conçu pour être responsive et adapté à un usage médical.
- Le backend est simple à étendre pour ajouter l'authentification, l'historique ou l'export CSV/PDF.
