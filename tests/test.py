import requests

a = requests.get('http://localhost:8686/eel.js')

with open('README.md', 'w') as file:
    file.write(a)

