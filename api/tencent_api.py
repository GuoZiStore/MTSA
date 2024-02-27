import json

from tencentcloud.common import credential
from tencentcloud.common.exception import TencentCloudSDKException
from tencentcloud.common.profile.client_profile import ClientProfile
from tencentcloud.common.profile.http_profile import HttpProfile
from tencentcloud.nlp.v20190408 import nlp_client, models

from config import tencent_client


def tencent_api(text):
    try:
        req = models.AnalyzeSentimentRequest()
        params = {
            "Text": text,
        }
        req.from_json_string(json.dumps(params))
        resp = tencent_client.AnalyzeSentiment(req)
        print(resp.to_json_string())
        return resp.Sentiment
    except TencentCloudSDKException as err:
        print(err)
