import requests
from bs4 import BeautifulSoup

import util.Article as articleClass

url = "https://www.scmp.com"
wbdata = requests.get(url).text

# Create Article DB
soup = BeautifulSoup(wbdata,'lxml')
news_titles = soup.select("div.article__title > a.article__link")

articles = list()
for n in news_titles:
    title = n.get_text()
    link = n.get("href")
    link = url + link
    artl = articleClass.article(title, link)
    articles.append(artl)

print(articles)
# Create Path DB
def decomposePath(articleList):
    for record in articleList:
        if  "/" in record:
            newList = record.split("/")
            decomposePath(newList)
