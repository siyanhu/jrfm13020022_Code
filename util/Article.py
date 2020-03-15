class article:
    title = ""
    link = ""
    raw_text = ""
    def __init__(self, article_title, article_link):
        self.title = article_title
        self.link = article_link

    def saveText(self, crawled):
        self.raw_text = crawled

    def __washText(self, textToWash):
        newText = ""
        # if "\n" in textToWash:

        return ""