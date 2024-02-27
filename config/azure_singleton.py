import boto3
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential


class AzureSingleton:
    _instance = None
    text_analytics_client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.text_analytics_client is None:
            language_key = ''
            language_endpoint = ''
            ta_credential = AzureKeyCredential(language_key)
            self.text_analytics_client = TextAnalyticsClient(endpoint=language_endpoint, credential=ta_credential)
