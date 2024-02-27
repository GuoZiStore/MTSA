from config import client


def azure_api(sentence):
    sentences = [sentence]
    result = client.analyze_sentiment(sentences, show_opinion_mining=True)
    doc_result = [doc.sentiment for doc in result if not doc.is_error]
    # print(doc_result)
    return doc_result[0]
