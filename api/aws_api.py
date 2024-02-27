from config import comprehend


def aws_api(text):
    res = comprehend.detect_sentiment(Text=text, LanguageCode='en')
    return res["Sentiment"]
