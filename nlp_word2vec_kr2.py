import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt

from gensim.models import Word2Vec
from gensim.models import KeyedVectors

#model.wv.save_word2vec_format('kor_w2v') # 모델 저장
loaded_model = KeyedVectors.load_word2vec_format("kor_w2v") # 모델 로드
print(loaded_model.wv.vectors.shape)
print(loaded_model.most_similar("황정민"))
