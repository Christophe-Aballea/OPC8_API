# API Credit Scoring Prêt à dépenser

## Description

Le projet déploie sur Heroku l'API de prédiction d'accord/refus de crédit de Prêt à dépenser.

## Fonctionnalités

- Retourne la probabilité de défaut de remboursement 
- Prédiction de l'accord ou du refus de crédit basé sur les données clients
- Retourne l'importance des caractéristiques locales dans la décision de crédit
- Interface Swagger pour la documentation et les tests de l'API
- Tests unitaires pour valider le bon fonctionnement de l'API
- Déploiement continu avec GitHub Actions

## Prérequis

- Python 3.12.4
- Flask
- LightGBM
- SHAP
- Gunicorn
- Requests

## Installation en local

1. Clonage du dépôt :
    ```bash
    git clone https://github.com/Christophe-Aballea/OPC8_API
    cd OPC8_API
    ```

2. Création et activation d'un environnement virtuel :
    ```bash
    python -m venv env
    source env/bin/activate
    ```

3. Installations des dépendances :
    ```bash
    pip install -r requirements.txt
    ```

## Utilisation

1. Lancement :
    ```bash
    gunicorn src.api:app
    ```

2. Accès via navigateur :  
- `http://127.0.0.1:5000` pour accéder à l'interface Swagger
- `http://127.0.0.1:5000/health` pour vérifier que l'API est fonctionnelle

## API Endpoints

### `GET /health`

- Description : Vérifie que l'API est opérationnelle.  

- Paramètres : Aucun.  

- Réponse (JSON) :  
    ```json
    {
      "message": "L'API est fonctionnelle.",  
      "documentation": "/swagger"  
    }
    ```

### `POST /predict`

- Description : Prédit la probabilité de défaut de remboursement et l'accord/refus de crédit.
- Body (JSON) :
    ```json
    {
      "columns": ["colonne1", "colonne2", "..."],
      "data": [
        ["valeur1", "valeur2", "..."]
      ]
    }
    ```

- Réponse (JSON) :
    ```json
    {
      "prediction_proba": [0.1234],
      "prediction_class": [1],
      "feature_names": ["feature1", "feature2", "..."],
      "feature_importance": [[0.01, 0.02, "..."]]
    }
    ```

## Tests

1. Vérifier que l'API est en cours d'exécution.  

2. Exécution des tests unitaires :
    ```bash
    python src/test.py
    ```  

3. Test de prédiction :
   Exécuter le notebook Jupyter `notebooks/test_API.ipynb` pour :
   - Sélectionner aléatoirement une demande de prêt parmi les données de test
   - Interroger l'API
   - Afficher l'ID du client, la probabilité de défaut de remboursement (score), l'accord ou le refus en fonction du seuil de classification
   - Afficher les 20 caractéristiques ayant le plus impacté la prédiction

## Déploiement

Le déploiement est automatisé avec GitHub Actions. Chaque `git push` déclenche le déploiement de l'API sur Heroku.  
Les tests unitaires sont intégrés au workflow GitHub Actions pour être exécutés sur l'API une fois déployée.  



## Structure du Projet

P08 - API/  
├── .github/  
│ └── workflows  
│   └── build_deploy.yml     # _Workflow Github Actions (déploiement et tests unitaires)_  
├── data/  
│ └── processed/   
│   ├── best_threshold.txt   # _Seuil de classification_   
│   ├── model.pkl            # _Modèle de classification LightGBM entraîné_  
│   └── preprocessor.pkl     # _Preprocessing et feature ingineering_  
├── notebooks/  
│ └── test_API.ipynb         # _Notebook de test de prédiction_  
├── src/  
│ ├── init.py  
│ ├── api.py                 # _Code de l'API_  
│ ├── test.py                # _Tests unitaires_  
├── .gitignore  
├── requirements.txt         # _Liste des versions des librairies python à installer_  
└── README.md                # _Documentation_  

## Auteurs

- Christophe ABALLEA - [Profil GitHub](https://github.com/Christophe-Aballea)

## Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](./LICENSE.md) pour plus de détails.
