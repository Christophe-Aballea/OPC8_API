import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
import lightgbm as lgb
from lightgbm import early_stopping
import numpy as np
import gc
import os
from sklearn.metrics import confusion_matrix
import shap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np
import re

FIRST_EXECUTION = False

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


def business_score(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel().tolist()
    score = 1 - ((10 * fn + fp) / (len(y_true) * 10))
    return score


def find_optimal_threshold(y_true, y_proba):
    thresholds = np.linspace(0, 1, 101)
    scores = [business_score(y_true, (y_proba > thresh).astype(int)) for thresh in thresholds]
    best_threshold = thresholds[np.argmax(scores)]
    best_score = max(scores)

    fine_thresholds = np.linspace(best_threshold - 0.01, best_threshold + 0.01, 21)
    fine_thresholds = [t for t in fine_thresholds if 0 <= t <= 1]
    fine_scores = [business_score(y_true, (y_proba > thresh).astype(int)) for thresh in fine_thresholds]
    best_threshold = fine_thresholds[np.argmax(fine_scores)]
    best_score = max(fine_scores)

    return best_threshold, best_score

def model(features, labels, n_folds=5):
    feature_names = list(features.columns)
    
    k_fold = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=50)
    out_of_fold = np.zeros(features.shape[0])
    best_thresholds = []
    valid_scores = []
    train_scores = []
    best_model = None
    best_valid_score = -np.inf
    
    for train_indices, valid_indices in k_fold.split(features, labels):
        train_features, train_labels = features.iloc[train_indices], labels[train_indices]
        valid_features, valid_labels = features.iloc[valid_indices], labels[valid_indices]
        
        model = lgb.LGBMClassifier(
            n_estimators=10000,
            objective='binary',
            class_weight='balanced',
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            n_jobs=-1,
            random_state=50,
            force_col_wise=True,
            verbosity=-1
        )
        
        model.fit(
            train_features, train_labels,
            eval_metric='auc',
            eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
            eval_names=['valid', 'train'],
            feature_name=feature_names,
            categorical_feature='auto',
            callbacks=[early_stopping(stopping_rounds=100)]
        )
        
        best_iteration = model.best_iteration_
        out_of_fold[valid_indices] = model.predict_proba(valid_features, num_iteration=best_iteration)[:, 1]
        
        best_threshold, _ = find_optimal_threshold(valid_labels, out_of_fold[valid_indices])
        best_thresholds.append(best_threshold)
        
        valid_pred_binary = (out_of_fold[valid_indices] > best_threshold).astype(int)
        train_pred_binary = (model.predict_proba(train_features, num_iteration=best_iteration)[:, 1] > best_threshold).astype(int)
        
        valid_score = business_score(valid_labels, valid_pred_binary)
        train_score = business_score(train_labels, train_pred_binary)
        
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        
        # Enregistrez le meilleur modèle basé sur la validation score
        if valid_score > best_valid_score:
            best_valid_score = valid_score
            best_model = model
        
        gc.enable()
        del train_features, valid_features
        gc.collect()
    
    best_threshold_overall, valid_business_score = find_optimal_threshold(labels, out_of_fold)
    best_thresholds.append(best_threshold_overall)
    
    valid_scores.append(valid_business_score)
    train_scores.append(np.mean(train_scores))
    
    fold_names = list(range(n_folds))
    fold_names.append('overall')
    
    metrics = pd.DataFrame({
        'fold': fold_names,
        'train': train_scores,
        'valid': valid_scores,
        'best_threshold': best_thresholds
    })
    
    # Obtenez les noms des features du modèle
    best_feature_names = best_model.booster_.feature_name()
    
    return metrics, best_threshold_overall, best_model, best_feature_names


# Emplacement des fichiers
base_dir = os.path.dirname(os.path.abspath(__file__))
app_train_path = os.path.join(base_dir, '..', 'data', 'raw', 'application_train.csv')
app_test_path = os.path.join(base_dir, '..', 'data', 'raw', 'application_test.csv')
preprocessor_path = os.path.join(base_dir, '..', 'data', 'processed', 'preprocessor.pkl')
model_path = os.path.join(base_dir, '..', 'data', 'processed', 'model.pkl')
best_threshold_path = os.path.join(base_dir, '..', 'data', 'processed', 'best_threshold.txt')
global_importance_path = os.path.join(base_dir, '..', 'data', 'processed', 'global_importance.pkl')
feature_names_path = os.path.join(base_dir, '..', 'data', 'processed', 'feature_names.pkl')

# Chargement des sonnées brutes
print("Chargement des données...")
app_train = pd.read_csv(app_train_path)
app_test = pd.read_csv(app_test_path)

# Préprocess
print("Preprocessing...")
preprocessor = Preprocessor()
preprocessor.fit(app_train.drop(columns=['TARGET', 'SK_ID_CURR']))

# Sauvegarde processor
print("Sauvegarde du processor...")
with open(preprocessor_path, 'wb') as f:
    pickle.dump(preprocessor, f)


app_train_preprocessed = preprocessor.transform(app_train.drop(columns=['TARGET', 'SK_ID_CURR']))

train_labels = app_train['TARGET']

# Entraînement du modèle et calcul du seuil de classification
print("Entraînement du modèle et calcul du seuil de classification...")
metrics, best_threshold, best_model, feature_names = model(app_train_preprocessed, train_labels)

print(f"Meilleur seuil de classification : {best_threshold}")

    
# Sauvegarde modèle
with open(model_path, 'wb') as model_file:
    pickle.dump(best_model, model_file)

# Sauvegarde seuil de classification
with open(best_threshold_path, 'w') as threshold_file:
    threshold_file.write(str(best_threshold))

# Sauvegarde features
with open(feature_names_path, 'wb') as features_file:
    pickle.dump(feature_names, features_file)

# Feature importance globale
print("Calcul feature importance globale...")
explainer = shap.TreeExplainer(best_model)
shap_values = explainer(app_train_preprocessed)

# Génération barplot de feature importance global SHAP
print("Génération du graphique...")
plt.figure(dpi=300, layout='constrained')
shap.plots.bar(shap_values, max_display=11, show=False)

# Personnalisation
bars = plt.gca().patches
annotations = plt.gca().texts

# Couleur des barres et annotations
for bar, annotation in zip(bars, annotations):
    bar.set_color('dodgerblue')
    annotation.set_color('dodgerblue')

# Modification du texte "Sum of X other features"
yticks = plt.gca().get_yticklabels()
for tick in yticks:
    if tick.get_text().startswith("Sum of"):
        tick.set_text("Autres caractéristiques")
plt.gca().set_yticklabels(yticks, fontsize=10)

# Taille xticks
xticks = plt.gca().get_xticklabels()
plt.gca().set_xticklabels(xticks, fontsize=10)

# Titre et légendes
plt.title("Top 10 des caractéristiques les plus impactantes\n(Moyenne sur l'historique des prêts)", fontsize=14, pad=15)
plt.xlabel("Influence moyenne (valeurs absolues SHAP)", fontsize=12)
plt.ylabel("Caractéristiques des prêts", fontsize=12)

# Sauvegarde
print("Sauvegarde du graphique...")
plt.savefig('global_importance_top10.svg')
plt.show()

# shap_values = np.array(explainer.shap_values(app_train_preprocessed[feature_names]))

# # Moyenne des valeurs absolues SHAP pour chaque feature
# global_importance = np.mean(np.abs(shap_values), axis=0)

# # Création d'un DataFrame
# global_importance_df = pd.DataFrame({'Feature': app_train_preprocessed.columns, 'Global importance': global_importance})

# # Sauvegarde feature importance globale
# print("Sauvegarde feature importance globale...")
# with open(global_importance_path, 'wb') as f:
#     pickle.dump(global_importance_df, f)
