import json
import urllib.request

url = 'http://127.0.0.1:5000/predict'
payload = {'landmarks': [0.1]*63}
data = json.dumps(payload).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
try:
    with urllib.request.urlopen(req, timeout=5) as resp:
        print('STATUS', resp.status)
        print(resp.read().decode())
except Exception as e:
    print('ERROR', e)
