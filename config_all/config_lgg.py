import numpy as np
import random


class Hyperparameter:
    #######################model settng#######################
    device = 'cuda'
    lr = 0.008
    beta1 = 0.9
    beta2 = 0.999
    weight_decay = 0.3471
    lambda_reg = 3e-3
    epoch_start = 1
    epoch_end = 38
    measure = 1
    verbose = 1

    #######################path setting#######################
    data_path = './data/lgg_train_test_cv5.pkl'
    model_save_path = './result_all/model_save/'
    pred_result_path = './result_all/pred_result/'
    datatype = 'lgg'
    ####### 'MtdfAtt'########
    model_name = 'MtdfAtt'

    # 'all' 'gene' 'cnv'
    modal = 'all'

    ####################### my model layer setting#######################
    LN_inputdim = 80
    LN_outputdim = 100

    LA_inputdim = LN_outputdim
    LA_outputdim = 60

    SelfAtt_inputdim = 2 * LN_outputdim
    SelfAtt_outputdim = 1

    R = 2  # Rank

    Lh1_2d_inputdim = LA_outputdim  # help="list_linear_h1_2d"
    Lh1_2d_outputdim = 30

    Lh2_inputdim = LA_outputdim  # help="list_linear_h2"
    Lh2_outputdim = Lh1_2d_outputdim

    Lam_inputdim = Lh2_outputdim  # help="linear_att_muatn"
    Lam_outputdim = R

    # 第二次塔克分解融合
    LL1f_inputdim = LN_outputdim  # help="list_linear_1_fusion"
    LL1f_outputdim = Lam_outputdim

    Lm_inputdim = R * Lam_outputdim  # help="linear_mlb" 这里有个cat80*2
    Lm_outputdim = Lh1_2d_inputdim

    L2f_inputdim = LN_outputdim  # help="linear_2_fusion"
    L2f_outputdim = Lm_outputdim

    Lcls_inputdim = SelfAtt_inputdim + 2 * Lh1_2d_outputdim  # help="linear_classif"
    Lcls_hiddendim = 1

    ####################### mlb setting #######################
    laml_inputdim = 60  # linear_att_mlb
    laml_outputdim = 30

    ####################### cat setting #######################
    lamc_inputdim = 120  # linear_att_mlb
    lamc_outputdim = 30

    ####################### noatt setting #######################
    lna_inputdim = 100
    lna_outputdim = 60

    cls_inputdim = 230
    cls_outputdim = 1

    ####################### mfh setting #######################
    mmdim = 100
    factor = 2
    l00_inputdim = 60  # linear0_0
    l00_outputdim = mmdim * 2

    lout_inputdim = mmdim * 2  # linear_out
    lout_outputdim = 30


HP = Hyperparameter()
