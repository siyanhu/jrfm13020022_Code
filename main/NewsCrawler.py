import requests
from bs4 import BeautifulSoup

import util.Article as articleClass
import structures.TreeStruct as tree

total_url = "https://www.scmp.com"
def articlesFromUrl(url):
    wbdata = requests.get(url).text
    # Create Article DB
    soup = BeautifulSoup(wbdata, 'lxml')
    news = soup.select("div.article__title > a.article__link")
    keywords = list()
    for new in news:
        link = n.get("href")
        keywords.append(link)
    return keywords

articles = articlesFromUrl(total_url)
# Create Path DB
total_mother = tree.node_str(total_url)
def decomposePath(mother, articleList):
    for record in articleList:
        if  "/" in record:
            newList = record.split("/")
            if len(newList) > 1:
                mother.appendChild(newList[0])
                # new_url =
            i = 0
            while i < len(newList):
                if (i == 0):
                    mother.appendChild(newList[i])

                i = i + 1