# -*- coding: utf-8 -*-
# @Time    : 2023/11/27 21:18
# @Author  : HuangHonghao
# @File    : __init __ .py.py
# @Description :
# 在其他地方调用单例模式
from config.aws_singleton import AwsSingleton
from config.azure_singleton import AzureSingleton
from config.cmu_singleton import CmuSingleton
from config.glove_model_singleton import GloveModelSingleton
from config.huawei_singleton import HuaWeiSingleton
from config.kw_model_singleton import KwModelSingleton
from config.nltk_sia_singleton import NltkSiaSingleton
from config.tencent_singleton import TencentSingleton

glove_model_instance = GloveModelSingleton()
glove_model_instance.get_instance()
glove_model = glove_model_instance.model

kw_model_instance = KwModelSingleton()
kw_model_instance.get_instance()
kw_model = kw_model_instance.model

# cmu_instance = CmuSingleton()
# cmu_instance.get_instance()
# cmu = cmu_instance.cmu
sia_instance = NltkSiaSingleton()
sia_instance.get_instance()
sia = sia_instance.sia

aws_instance = AwsSingleton()
aws_instance.get_instance()
comprehend = aws_instance.comprehend

azure_instance = AzureSingleton()
azure_instance.get_instance()
client = azure_instance.text_analytics_client

tencent_instance = TencentSingleton()
tencent_instance.get_instance()
tencent_client = tencent_instance.client

huawei_instance = HuaWeiSingleton()
huawei_instance.get_instance()
huawei_client = huawei_instance.client
