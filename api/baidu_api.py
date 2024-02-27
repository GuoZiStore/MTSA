# -*- coding: utf-8 -*-
# @Time    : 2023/10/3 21:38
# @Author  : HuangHonghao
# @File    : baidu_api.py
# @Description : 百度API配置
import time

from config.baidu_api_config import AipNlpSingleton
from res.baidu_api_res import BaiduApiRes


def baidu_api(text):
    """"""
    # time.sleep(0.5)
    client = AipNlpSingleton()
    result = client.sentimentClassify(text)
    # print(result)
    return BaiduApiRes.from_dict(result)
