import numpy as np
import pandas as pd
import scipy as sc
import spacy
import torch
import GPUtil

import os
import json
import warnings

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


spacy.util.fix_random_seed(0)
is_using_gpu = spacy.prefer_gpu()
if is_using_gpu:
  torch.set_default_tensor_type("torch.cuda.FloatTensor")
  print("GPU Usage")
  GPUtil.showUtilization()


warnings.filterwarnings("ignore")
#model = SentenceTransformer('bert-base-nli-mean-tokens')

model = SentenceTransformer('bert-large-nli-mean-tokens')

print(len(triggers))
print(triggers)
#trigger_embeddings = model.encode(triggers)