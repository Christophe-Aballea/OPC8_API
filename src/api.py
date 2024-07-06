import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import shap
from src.utils.preprocessing import Preprocessor

app = Flask(__name__)

# Emplacement des fichiers
base_dir = os.path.dirname(os.path.abspath(__file__))
preprocessor_path = os.path.join(base_dir, '..', 'data', 'processed', 'preprocessor.pkl')
model_path = os.path.join(base_dir, '..', 'data', 'processed', 'model.pkl')

# Chargement preprocessor et model
with open(model_path, 'rb') as model_file:
    best_model = pickle.load(model_file)

with open(preprocessor_path, 'rb') as f:
    preprocessor = pickle.load(f)

# Initialisation SHAP explainer
explainer = shap.TreeExplainer(best_model)


# Route /predict
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data['data'], columns=data['columns'])
    df = df.replace({None: np.nan})
    processed_data = preprocessor.transform(df)
    prediction_proba = best_model.predict_proba(processed_data)[:, 1]
    shap_values = explainer.shap_values(processed_data)
    return jsonify({
        'prediction_proba': prediction_proba.tolist(),
        'feature_names': processed_data.columns.tolist(),
        'feature_importance': shap_values.tolist()
    })


if __name__ == '__main__':
    app.run(debug=True) 
