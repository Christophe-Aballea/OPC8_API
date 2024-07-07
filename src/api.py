import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pickle
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

# Initialisation Flask
app = Flask(__name__)
api = Api(app, version='1.0', title='Credit Scoring API',
          description='A simple Credit Scoring API',
          )

ns = api.namespace('predict', description='Prediction operations')

# Modèle de données pour Swagger
predict_model = api.model('PredictionModel', {
    'data': fields.List(fields.List(fields.Raw), required=True, description='List of data values'),
    'columns': fields.List(fields.String, required=True, description='List of column names')
})


class Preprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.le_dict = {}
        self.columns = None

    def fit(self, X, y=None):
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
        for col, le in self.le_dict.items():
            X[col] = le.transform(X[col])
        X = pd.get_dummies(X)
        X['DAYS_EMPLOYED_ANOM'] = X["DAYS_EMPLOYED"] == 365243
        X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].replace({365243: np.nan})
        X['DAYS_EMPLOYED'] = abs(X['DAYS_EMPLOYED'])
        X['DAYS_BIRTH'] = abs(X['DAYS_BIRTH'])
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

# Initialisation SHAP explainer
explainer = shap.TreeExplainer(best_model)


@app.route('/')
def index():
    return jsonify({
        "message": "Welcome to the Credit Scoring API.",
        "documentation": "/swagger/"
    })


# Route /predict
@ns.route('/')
class Prediction(Resource):
    @ns.expect(predict_model)
    def post(self):
        data = request.json
        df = pd.DataFrame(data['data'], columns=data['columns'])
        df = df.replace({None: np.nan})
        processed_data = preprocessor.transform(df)
        prediction_proba = best_model.predict_proba(processed_data)[:, 1]
        prediction_class = (prediction_proba >= best_threshold) * 1
        shap_values = explainer.shap_values(processed_data)
        return jsonify({
            'prediction_proba': prediction_proba.tolist(),
            'prediction_class': [prediction_class],
            'feature_names': processed_data.columns.tolist(),
            'feature_importance': shap_values.tolist()
        })


if __name__ == '__main__':
    app.run(debug=True)
