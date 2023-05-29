import requests
import json

url = 'http://localhost:5000/api/'

# Change the value of experience that we want to test

payload = {
    'exp'
}

r = requests.post(url,json={'_rxn_M_acid': 5,'_rxn_M_organic': 7,'_stoich_mmol_org' : 8
               ,'_rxn_M_inorganic': 6})

