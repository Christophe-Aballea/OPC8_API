import os
import re
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import pickle
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

pd.set_option('future.no_silent_downcasting', True)

# Initialisation Flask
app = Flask(__name__)

class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le_dict = {}
        self.columns = None
        
    def fit(self, X, y=None):
        # Fit du label encoder pour les features catégorielles 
        for col in X:
            if X[col].dtype == 'object' and len(list(X[col].unique())) <= 2:
                le = LabelEncoder()
                le.fit(X[col])
                self.le_dict[col] = le
        X_transformed = self._transform(X)
        self.columns = X_transformed.columns
        return self

    def transform(self, X):
        X_transformed = self._transform(X)
        X_transformed = X_transformed.reindex(columns=self.columns, fill_value=0)
        return X_transformed
    
    def _transform(self, X):
        X = X.copy()
        
        # Label encoding
        for col, le in self.le_dict.items():
            X[col] = le.transform(X[col])
        
        # One-hot encoding
        X = pd.get_dummies(X)

        # Nombres de jours négatifs -> positifs
        X['DAYS_REGISTRATION'] = abs(X['DAYS_REGISTRATION'])
        X['DAYS_ID_PUBLISH'] = abs(X['DAYS_ID_PUBLISH'])
              
        # Dates en anomalies
        X['YEARS_EMPLOYED_ANOM'] = X["DAYS_EMPLOYED"] == 365243
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace({365243: np.nan})
  
        # Nombre de jours -> années
        X['YEARS_EMPLOYED'] = abs(X['DAYS_EMPLOYED'] / 365.25)
        X['YEARS_BIRTH'] = abs(X['DAYS_BIRTH'] / 365.25)
        X['YEARS_LAST_PHONE_CHANGE'] = abs(X['DAYS_LAST_PHONE_CHANGE'] / 365.25)
      
        X = X.drop(columns=['DAYS_EMPLOYED', 'DAYS_BIRTH', 'DAYS_LAST_PHONE_CHANGE'])
        
        # Feature engineering
        X['CREDIT_INCOME_PERCENT'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
        X['ANNUITY_INCOME_PERCENT'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
        X['CREDIT_TERM'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
        X['DAYS_EMPLOYED_PERCENT'] = X['YEARS_EMPLOYED'] / X['YEARS_BIRTH']

        X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))

        return X


# Déserialisation preprocessor
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
threshold_path = os.path.join(base_dir, '..', 'data', 'processed', 'best_threshold.txt')

# Chargement modèle
with open(model_path, 'rb') as model_file:
    best_model = pickle.load(model_file)

# Chargement preprocessor
with open(preprocessor_path, 'rb') as f:
    preprocessor = custom_load(f, {'Preprocessor': Preprocessor})

# Récupération du seuil de classification
with open(threshold_path, 'r') as threshold_file:
    best_threshold = float(threshold_file.read())


# Route pour test
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'message': 'L\'API est fonctionnelle',
        'documentation': '/swagger'
    })
    
# Route '/predict'
@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame(data['data'], columns=data['columns'])
        df = df.replace({None: pd.NA})
        processed_data = preprocessor.transform(df)
        prediction_proba = best_model.predict_proba(processed_data)[:, 1]
        prediction_class = (prediction_proba >= best_threshold) * 1
        return jsonify({
            'prediction_proba': prediction_proba.tolist(),
            'prediction_class': prediction_class.tolist(),
        })
    except Exception as e:
        print("Erreur pendant la prédiction:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
