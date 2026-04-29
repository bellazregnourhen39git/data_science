from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from pathlib import Path
import uuid

from model_interface import (
    load_models,
    get_dashboard_metrics,
    predict_patient,
    init_history_db,
    save_prediction_history,
    get_prediction_history,
    export_history_csv
)

app = Flask(__name__)
CORS(app)

BASE_DIR = Path(__file__).resolve().parent
DATABASE_PATH = BASE_DIR / 'history.db'
models = load_models()
init_history_db(DATABASE_PATH)

AUTH_USERS = {
    'medecin': 'securite123'
}
SESSIONS = {}


def _get_token_from_request():
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header.replace('Bearer ', '').strip()
    return None


def _get_current_user():
    token = _get_token_from_request()
    if token and token in SESSIONS:
        return SESSIONS[token]
    return None


@app.route('/api/health', methods=['GET'])
def health():
    return jsonify(status='ok', message='Backend opérationnel')


@app.route('/api/login', methods=['POST'])
def login():
    data = request.json or {}
    username = data.get('username')
    password = data.get('password')

    if not username or not password:
        return jsonify(error='Nom d’utilisateur et mot de passe requis'), 400
    if AUTH_USERS.get(username) != password:
        return jsonify(error='Nom d’utilisateur ou mot de passe incorrect'), 401

    token = str(uuid.uuid4())
    SESSIONS[token] = username
    return jsonify(token=token, username=username)


@app.route('/api/metrics', methods=['GET'])
def metrics():
    return jsonify(get_dashboard_metrics())


@app.route('/api/predict', methods=['POST'])
def predict():
    user = _get_current_user()
    if not user:
        return jsonify(error='Authentification requise'), 401

    payload = request.json
    if not payload:
        return jsonify(error='Aucune donnée reçue'), 400

    try:
        result = predict_patient(models, payload)
        save_prediction_history(DATABASE_PATH, user, payload, result)
        return jsonify(result)
    except Exception as exc:
        return jsonify(error=str(exc)), 400


@app.route('/api/history', methods=['GET'])
def history():
    user = _get_current_user()
    if not user:
        return jsonify(error='Authentification requise'), 401

    history_data = get_prediction_history(DATABASE_PATH, user)
    return jsonify(history_data)


@app.route('/api/export/csv', methods=['GET'])
def export_csv():
    user = _get_current_user()
    if not user:
        return jsonify(error='Authentification requise'), 401

    csv_buffer = export_history_csv(DATABASE_PATH, user)
    return send_file(
        csv_buffer,
        mimetype='text/csv',
        as_attachment=True,
        download_name=f'history_{user}.csv'
    )


@app.route('/api/logout', methods=['POST'])
def logout():
    token = _get_token_from_request()
    if token and token in SESSIONS:
        del SESSIONS[token]
    return jsonify(message='Déconnecté')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
