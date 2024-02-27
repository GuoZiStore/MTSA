import boto3


class AwsSingleton:
    _instance = None
    comprehend = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.comprehend is None:
            self.comprehend = boto3.client(service_name='comprehend', region_name='us-east-1',
                                           aws_access_key_id='',
                                           aws_secret_access_key='')
