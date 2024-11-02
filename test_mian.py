import os
import logging
import numpy as np
import pickle

import torch
from model_att import *
from train_test import train, test

### 1. Initializes parser and device
type_ = 'lgg'
device = 'cuda'
print("Using device:", device)

data_cv_path = './data/%s_train_test_cv5.pkl' % type_
print("Loading %s" % data_cv_path)
data_cv = pickle.load(open(data_cv_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
results = []
acc = []

for k, data in data_cv_splits.items():
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
    print("*******************************************")
    load_path = './result_all/model_save/%s_model_%s.pt' % (type_, k)
    model_ckpt = torch.load(load_path, map_location=device)
    #### Loading Env
    model_state_dict = model_ckpt['model_state_dict']

    model = MtdfAtt().to('cuda')
    model.load_state_dict(model_state_dict)

    loss_test, cindex_test, pvalue_test, pred_test = test(model, data, 'test')

    print("[Final] Apply model to testing set: C-Index: %.10f" % (cindex_test))
    results.append(cindex_test)


print('Split Results:', results)
print("Average_C_index: %.3f" % np.array(results).mean(), " std: %.3f" % np.std(results, ddof=0))

