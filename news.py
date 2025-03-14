import requests
import pandas as pd


# news api-key:
# 1270f9ec3e2a40f3a58f67bba8b716f4


def call_news_api():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "bitcoin", "apiKey": "1270f9ec3e2a40f3a58f67bba8b716f4"
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        print(data)
        articles = data.get("articles", [])

        df = pd.DataFrame(articles)
        df.to_excel("news_articles.xlsx", index=False)
        df.to_csv("news_articles.csv", index=False)
        print("Data saved to news_articles in both xlsx and csv format")
    else:
        print(f"Error: {response.status_code}", response.text)


if __name__ == "__main__":
    call_news_api()

