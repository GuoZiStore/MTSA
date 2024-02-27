from aip import AipNlp

APP_ID = ''
API_KEY = ''
SECRET_KEY = ''


def singleton(cls):
    instances = {}

    def getinstance():
        if cls not in instances:
            instances[cls] = cls(APP_ID, API_KEY, SECRET_KEY)
        return instances[cls]

    return getinstance


@singleton
class AipNlpSingleton(AipNlp):
    pass
