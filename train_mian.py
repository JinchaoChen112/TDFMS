import os
import numpy as np
import pickle
import pandas as pd

from data_loaders import *
from train_test import train, test

np.set_printoptions(suppress=True)

from config_all.config_lgg import HP

### 3. Sets-Up Main Loop
print('######################################################################')
print('The current dataset is: ', HP.data_path)
print('The current model is: ', HP.model_name)
print('The current mode is: ', HP.modal)
print('######################################################################')
data_cv = pickle.load(open(HP.data_path, 'rb'))
data_cv_splits = data_cv['cv_splits']
average_results = []
os_time = []
os_status = []
risk_pred = []
label_pred = []
id_ = []

for k, data in data_cv_splits.items():
    print("*******************************************")
    print("************** SPLIT (%d/%d) **************" % (k, len(data_cv_splits.items())))
    print("*******************************************")

    model, optimizer, metric_logger = train(data)

    loss_train, cindex_train, pvalue_train, pred_train = test(model, data, 'train')
    loss_test, cindex_test, pvalue_test, pred_test = test(model, data, 'test')

    print("[Final] Apply model to testing set: C-Index: %.10f" % (cindex_test))
    print("[Final] Apply model to testing set: P-Value: %.10e" % (pvalue_test))
    average_results.append(cindex_test)

    model_state_dict = model.state_dict()
    torch.save({
        'epoch': HP.epoch_end,
        'data': data,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metric_logger},
        os.path.join(HP.model_save_path, '%s_model_%s.pt' % (HP.datatype, k))
    )

    result = pd.DataFrame(
        {'os_time': pred_test[0], 'os_status': pred_test[1], 'risk_pred': pred_test[2], 'id': pred_test[3]})
    result.to_csv(HP.pred_result_path + '%s_%s_Fold_pred_result.csv' % (HP.datatype, k))

    os_time.extend(pred_test[0])
    os_status.extend(pred_test[1])
    risk_pred.extend(pred_test[2])
    id_.extend(pred_test[3])

result_all = pd.DataFrame({'os_time': os_time, 'os_status': os_status, 'risk_pred': risk_pred, 'id': id_})
result_all.to_csv(HP.pred_result_path + '%s_all_pred_result.csv' % HP.datatype)

c_index_all = pd.DataFrame({'C_index': average_results})
c_index_all.to_csv('./result_all/pred_result/%s_all_c_index_result.csv' % HP.datatype)

print('Split Average Results:', average_results)
print("Average_C_index: %.3f" % np.array(average_results).mean(), " std: %.3f" % np.std(average_results, ddof=0))
