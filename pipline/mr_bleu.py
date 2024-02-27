from config.generation_rules import model_predict_mrs
from implement.similar_util import mr_self_bleu

mr_self_bleu(task="sst2", rules=model_predict_mrs, text_index=1, threshold=0.8)
