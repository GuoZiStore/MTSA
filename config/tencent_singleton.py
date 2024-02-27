from tencentcloud.common import credential
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.nlp.v20190408 import nlp_client


class TencentSingleton:
    _instance = None
    client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.client is None:
            cred = credential.Credential("", "")
            http_profile = HttpProfile()
            http_profile.endpoint = "nlp.tencentcloudapi.com"
            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile
            self.client = nlp_client.NlpClient(cred, "", client_profile)
