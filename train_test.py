from tqdm import tqdm
import torch.backends.cudnn as cudnn
from model_att import *
from torch.utils.data import DataLoader
from data_loaders import Dataset_loader
from utils import CoxLoss, regularize_weights, CIndex_lifeline, cox_log_rank, count_parameters
import gc
from config_all.config_lgg import HP

def train(data):
    cudnn.deterministic = True
    torch.cuda.manual_seed_all(666)
    torch.manual_seed(66)

    #############
    if HP.model_name == 'MtdfAtt':
        model = MtdfAtt().to(HP.device)
        print('MtdfAtt')
    else:
        print('no model')
    #############
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.lr, betas=(HP.beta1, HP.beta2), weight_decay=HP.weight_decay)

    formatted_param_count = count_parameters(model)
    print("Number of Trainable Parameters: %s M" % formatted_param_count)


    custom_data_loader = Dataset_loader(data, split='train')

    train_loader = DataLoader(dataset=custom_data_loader, batch_size=len(custom_data_loader), shuffle=True, drop_last=False)
    metric_logger = {'train': {'loss': [], 'pvalue': [], 'cindex': []},
                     'test': {'loss': [], 'pvalue': [], 'cindex': []}}
    c_index_best = 0

    for epoch in tqdm(range(HP.epoch_start, HP.epoch_end + 1)):
        model.train()
        risk_pred_all, censor_all, survtime_all, id_all = np.array([]), np.array([]), np.array([]), np.array([])
        loss_epoch = 0
        gc.collect()
        for batch_idx, (gene, cnv, censor, survtime, id_) in enumerate(train_loader):
            censor = censor.to(HP.device)

            ##############
            if HP.modal == 'all':
                pred = model(gene.to(HP.device), cnv.to(HP.device))
                print('all')
            ##############
            loss_cox = CoxLoss(survtime, censor, pred)
            loss_reg = regularize_weights(model=model)
            loss = loss_cox + HP.lambda_reg * loss_reg

            loss_epoch += loss.data.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.empty_cache()

            risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
            censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
            survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
            id_all = np.concatenate((id_all, id_))

        if HP.measure or epoch == (HP.epoch_end - 1):
            loss_epoch /= len(custom_data_loader)

            cindex_epoch = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
            pvalue_epoch = cox_log_rank(risk_pred_all, censor_all, survtime_all)
            loss_test, cindex_test, pvalue_test, pred_test = test(model, data, 'test')

            metric_logger['train']['loss'].append(loss_epoch)
            metric_logger['train']['cindex'].append(cindex_epoch)
            metric_logger['train']['pvalue'].append(pvalue_epoch)

            metric_logger['test']['loss'].append(loss_test)
            metric_logger['test']['cindex'].append(cindex_test)
            metric_logger['test']['pvalue'].append(pvalue_test)

            if cindex_test > c_index_best:
                c_index_best = cindex_test
            if HP.verbose > 0:
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}'.format('Train', loss_epoch, 'C-Index', cindex_epoch))
                print('[{:s}]\t\tLoss: {:.4f}, {:s}: {:.4f}\n'.format('Test', loss_test, 'C-Index', cindex_test))

    return model, optimizer, metric_logger


def test(model, data, split):
    model.eval()
    custom_data_loader = Dataset_loader(data, split)

    test_loader = DataLoader(dataset=custom_data_loader, batch_size=len(custom_data_loader), shuffle=True, drop_last=False)
    risk_pred_all, censor_all, survtime_all, id_all = np.array([]), np.array([]), np.array([]), np.array([])
    loss_test = 0

    for batch_idx, (gene, cnv, censor, survtime, id_) in enumerate(test_loader):
        censor = censor.to(HP.device)

        #################
        if HP.modal == 'all':
            pred = model(gene.to(HP.device), cnv.to(HP.device))
        #################
        loss_cox = CoxLoss(survtime, censor, pred)
        loss_reg = regularize_weights(model=model)
        loss = loss_cox + HP.lambda_reg * loss_reg
        loss_test += loss.data.item()

        risk_pred_all = np.concatenate((risk_pred_all, pred.detach().cpu().numpy().reshape(-1)))
        censor_all = np.concatenate((censor_all, censor.detach().cpu().numpy().reshape(-1)))
        survtime_all = np.concatenate((survtime_all, survtime.detach().cpu().numpy().reshape(-1)))
        id_all = np.concatenate((id_all, id_))


    ###################################################
    # ==== Measuring Test Loss, C-Index, P-Value ==== #
    ###################################################
    loss_test /= len(custom_data_loader)
    cindex_test = CIndex_lifeline(risk_pred_all, censor_all, survtime_all)
    pvalue_test = cox_log_rank(risk_pred_all, censor_all, survtime_all)
    pred_test = [survtime_all, censor_all, risk_pred_all, id_all]
    return loss_test, cindex_test, pvalue_test, pred_test
