from huaweicloudsdknlp.v2 import RunSentimentRequest, HWCloudSentimentReq

from config import huawei_client
from huaweicloudsdkcore.exceptions import exceptions


def huawei_api(text):
    try:
        request = RunSentimentRequest()
        request.body = HWCloudSentimentReq(
            content=text
        )
        response = huawei_client.run_sentiment(request)
        return response.result.label
    except exceptions.ClientRequestException as e:
        print(e.status_code)
        print(e.request_id)
        print(e.error_code)
        print(e.error_msg)

