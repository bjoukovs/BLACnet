import requests
from bs4 import BeautifulSoup

url = "https://twitter.com/BradyHaran/status/1072113227847397376"

r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')

tweets = soup.findAll('p', class_='tweet-text')

for tweet in tweets:
    print(tweet.text.encode("utf-8"))
