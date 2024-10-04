import requests


if __name__ == '__main__':
    comment = {'teams': ['Arsenal', 'Liverpool']}

    url = 'http://0.0.0.0:8000/predict'
    response = requests.post(url, json=comment)
    print(response.json())
