from huaweicloudsdkcore.auth.credentials import BasicCredentials
from huaweicloudsdknlp.v2.region.nlp_region import NlpRegion
from huaweicloudsdkcore.exceptions import exceptions
from huaweicloudsdknlp.v2 import *


class HuaWeiSingleton:
    _instance = None
    client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.get_instance()
        return cls._instance

    def get_instance(self):
        if self.client is None:
            ak = ""
            sk = ""
            credentials = BasicCredentials(ak, sk)
            self.client = NlpClient.new_builder().with_credentials(credentials).with_region(
                NlpRegion.value_of("cn-north-4")).build()
