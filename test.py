import requests
import json

url = 'http://localhost:5000/process'
files = {'file': open('document.pdf', 'rb')}

try:
    response = requests.post(url, files=files)
    response.raise_for_status()
    results = response.json()
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
except requests.exceptions.RequestException as e:
    print("Error:", e)
    if hasattr(e.response, 'json'):
        print("Server error details:", e.response.json())