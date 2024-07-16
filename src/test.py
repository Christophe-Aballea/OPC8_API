import unittest
import requests
import json
import pandas as pd

class TestCreditScoringAPI(unittest.TestCase):
    BASE_URL = "https://failurescore-bc9f53f25e58.herokuapp.com"

    def test_predict(self):
        # Exemple de données pour le test
        test_data = {
            "data": [
                ["Cash loans", "F", "N", "Y", 0, 225000.0, 238981.5, 12330.0, 162000.0, "Unaccompanied", "Pensioner", "Secondary / secondary special", "Married", "Rented apartment", 0.025164, -22202, 365243, -5539.0, -4161, None, 1, 0, 0, 1, 0, 0, None, 2.0, 2, 2, "FRIDAY", 11, 0, 0, 0, 0, 0, 0, "XNA", None, 0.2254321061189867, 0.1595195404777181, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, 2.0, 1.0, 2.0, 1.0, -71.0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.0]
            ]
        }
        
        # Convertir les données au format JSON
        json_data = json.dumps(test_data)
        
        # Faire une requête POST à l'API
        response = requests.post(f"{self.BASE_URL}/predict/", headers={'Content-Type': 'application/json'}, data=json_data)
        
        # Vérifier le statut de la réponse
        self.assertEqual(response.status_code, 200)
        
        # Vérifier la structure de la réponse JSON
        response_json = response.json()
        self.assertIn('prediction_proba', response_json)
        self.assertIn('prediction_class', response_json)
        # self.assertIn('feature_names', response_json)
        
    def test_health_check(self):
        # Faire une requête GET à la route de vérification de l'état de santé
        response = requests.get(f"{self.BASE_URL}/health")

        # Vérifier le statut de la réponse
        self.assertEqual(response.status_code, 200)

        # Vérifier le contenu de la réponse
        try:
            response_json = response.json()
        except ValueError:
            self.fail("Response is not in JSON format")

        self.assertIn('message', response_json)
        self.assertIn('documentation', response_json)
        self.assertEqual(response_json['message'], "L'API est fonctionnelle")


if __name__ == '__main__':
    unittest.main()
