import os
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import numpy as np
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np

app = Flask(__name__)

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le_dict = {}
        
    def fit(self, X, y=None):
        # Fit du label encoder pour les features catégorielles 
        for col in X:
            if X[col].dtype == 'object' and len(list(X[col].unique())) <= 2:
                le = LabelEncoder()
                le.fit(X[col])
                self.le_dict[col] = le
        
        return self

    def transform(self, X):
        X = X.copy()
        
        # Label encoding
        for col, le in self.le_dict.items():
            X[col] = le.transform(X[col])
        
        # One-hot encoding
        X = pd.get_dummies(X)
        
        # Alignement train et test
        if hasattr(self, 'columns'):
            X = X.reindex(columns=self.columns, fill_value=0)
        else:
            self.columns = X.columns
        
        # Dates en anomalies
        X['DAYS_EMPLOYED_ANOM'] = X["DAYS_EMPLOYED"] == 365243
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace({365243: np.nan})
        X['DAYS_EMPLOYED'] = abs(X['DAYS_EMPLOYED'])
        X['DAYS_BIRTH'] = abs(X['DAYS_BIRTH'])
        
        # Feature engineering
        X['CREDIT_INCOME_PERCENT'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['ANNUITY_INCOME_PERCENT'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['CREDIT_TERM'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['DAYS_EMPLOYED_PERCENT'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
        
        return X

def custom_load(pickle_file, class_dict):
    class CustomUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name in class_dict:
                return class_dict[name]
            return super().find_class(module, name)
    
    return CustomUnpickler(pickle_file).load()


# Emplacement des fichiers
base_dir = os.path.dirname(os.path.abspath(__file__))
preprocessor_path = os.path.join(base_dir, '..', 'data', 'processed', 'preprocessor.pkl')
model_path = os.path.join(base_dir, '..', 'data', 'processed', 'model.pkl')

def custom_load(path):
    with open(path, 'rb') as file:
        return pickle.load(file, fix_imports=True, encoding="bytes", errors="strict")

# Chargement preprocessor et model
with open(model_path, 'rb') as model_file:
    best_model = pickle.load(model_file)

with open(preprocessor_path, 'rb') as f:
    preprocessor = custom_load(f, {'Preprocessor': Preprocessor})

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
