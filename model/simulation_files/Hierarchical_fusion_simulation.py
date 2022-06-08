import os, sys, time
import pickle as pkl
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from importlib import reload
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import torch
from torch import optim, nn
import torch.utils.data as Data
from torch.nn import functional as F
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def seed_torch(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)#as reproducibility docs
    torch.manual_seed(seed)# as reproducibility docs
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False# as reproducibility docs
    torch.backends.cudnn.deterministic = True# as reproducibility docs
    
target_encoder = LabelEncoder()
onehot_encoder = OneHotEncoder(sparse=False)

def load_data(impute_method = 'Mean-Mode', target = 'NACCAD3'):
    uds = pd.read_csv("../data/data_imputed/{}/uds.csv".format(impute_method))
    uds['datetime'] = pd.to_datetime(uds['datetime'])
    uds = uds.dropna(subset=[target, 'EDUC'])
    
    print("Target Distribution: {}\n".format(Counter(uds[target])))
    uds[target] = target_encoder.fit_transform(uds[target])
    onehot_encoder.fit(uds[target].values.reshape(-1, 1))

    mri = pd.read_csv("../data/data_imputed/{}/mri.csv".format(impute_method))
    mri['datetime'] = pd.to_datetime(mri['datetime'])
    
    csf = pd.read_csv("../data/data_imputed/{}/csf.csv".format(impute_method))
    return uds, mri, csf

uds_dict = pd.read_csv("../data/data_dictionary/uds_feature_dictionary_cleaned.csv")
mri_dict = pd.read_csv("../data/data_dictionary/mri_feature_dictionary_cleaned.csv") 

uds_drop_columns = ['NACCID', 'NACCADC', 'NACCVNUM', 'datetime', 'NACCUDSD', 'NACCALZP', 'NACCAD3', 'NACCAD5']
mri_drop_columns = ['NACCID', 'NACCVNUM', 'datetime', 'datetime_UDS', 'timediff', 'within-a-year']
csf_drop_columns = ['NACCID', 'CSFABMD', 'CSFTTMD', 'CSFPTMD']

target = 'NACCAD3'
uds, mri, csf = load_data(target = target)


from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

import warnings
warnings.filterwarnings('ignore')

def obtain_metrics(ytr, ytr_hat, yte, yte_hat, confusion_metrix = False, df_type=None, seed=None):
    if confusion_metrix:
        print(metrics.confusion_matrix(ytr, ytr_hat))
        print(metrics.confusion_matrix(yte, yte_hat))
        
    acctr = metrics.accuracy_score(ytr, ytr_hat)
    f1tr_macro = metrics.f1_score(ytr, ytr_hat, average='macro')
    f1tr_micro = metrics.f1_score(ytr, ytr_hat, average='micro')
    
    accte = metrics.accuracy_score(yte, yte_hat)
    f1te_macro = metrics.f1_score(yte, yte_hat, average='macro')
    f1te_micro = metrics.f1_score(yte, yte_hat, average='micro')
 
    metrics_df = pd.DataFrame.from_dict({"Train": {"Acc": acctr, "F1-macro": f1tr_macro, "F1-micro": f1tr_micro}, 
                                         "Test": {"Acc": accte, "F1-macro": f1te_macro, "F1-micro": f1te_micro}}, 
                                         orient='Index')
    metrics_df = metrics_df.round(3)
    if df_type is not None:
        metrics_df['Type'] = df_type
    if seed is not None:
        metrics_df['Seed'] = seed
    return metrics_df
    
def print_summary_clf(clf, Xtr, ytr, Xte, yte, confusion_metrix = False, df_type = None, seed=None):

    metrics_df = obtain_metrics(ytr, clf.predict(Xtr), yte, clf.predict(Xte), confusion_metrix=confusion_metrix)
    return metrics_df


def find_index(a, b):
    overlap_ppl = set(a['NACCID']).intersection(b['NACCID'])
    aidx = a[a['NACCID'].isin(overlap_ppl)].reset_index().sort_values('NACCID')['index'].values
    bidx = b[b['NACCID'].isin(overlap_ppl)].reset_index().sort_values('NACCID')['index'].values
    return aidx, bidx

def find_all_index(a, b, c):
    overlap_ppl = set(a['NACCID']).intersection(b['NACCID']).intersection(c['NACCID'])
    aidx = a[a['NACCID'].isin(overlap_ppl)].reset_index().sort_values('NACCID')['index'].values
    bidx = b[b['NACCID'].isin(overlap_ppl)].reset_index().sort_values('NACCID')['index'].values
    cidx = c[c['NACCID'].isin(overlap_ppl)].reset_index().sort_values('NACCID')['index'].values
    return aidx, bidx, cidx

class PredModel(torch.nn.Module):
    def __init__(self, layers, seed=48):
        super(PredModel, self).__init__()
        seed_torch(seed)
        linear_list = [nn.Linear(layers[0], layers[1])]
        for i in range(2, len(layers)):
            linear_list.append(nn.ReLU())
            linear_list.append(nn.Dropout(0.1))
            linear_list.append(nn.Linear(layers[i-1], layers[i]))   
        self.layers = nn.Sequential(*linear_list)
    def forward(self, X):
        return self.layers(X)
    def predict_proba(self, X):
        return nn.Softmax()(self.layers(X)).detach().cpu().numpy()
    def predict(self, X):
        return nn.Softmax()(self.layers(X)).argmax(1).cpu().numpy()
    def loss(self, X, y):
        y_proba = self.forward(X)
        return nn.CrossEntropyLoss()(y_proba, y)
    
def train_pred(clf, Xtr, ytr, Xte, yte, display_intvl=250):
    optimizer = torch.optim.Adam(clf.parameters(), lr=1e-4, weight_decay=1e-5)
    last_loss, loss = 0, 0
    for epoch in range(1000):
        loss =  clf.loss(Xtr, ytr)
        loss.backward()
        optimizer.step()
        if (epoch % display_intvl == 0 or torch.abs(last_loss - loss) < 1e-9):
            acc, f1_macro, f1_micro = print_summary_clf(clf, Xtr, ytr, Xte, yte).loc["Train",]
            print('Pred Epoch {}:  Loss: {:.3f}\tAcc-{:.3f}%\tF1_macro-{:.3f}\tF1_micro-{:.3f}'.format(
                epoch, loss.item(), acc*100, f1_macro, f1_micro))
        if torch.abs(last_loss - loss) < 1e-9:
            break
        last_loss = loss
    return clf


def prepare_autoencoder_data(ppl_list, original=True):
    U = uds[uds['NACCID'].isin(ppl_list)][['NACCID', 'NACCAD3'] + Ucol]
    Utr = torch.tensor(U[Ucol].to_numpy()).float().to(device)
 
    M = mri[mri['NACCID'].isin(ppl_list)][['NACCID'] + Mcol]
    Mtr = torch.tensor(M[Mcol].to_numpy()).float().to(device)

    C = csf[csf['NACCID'].isin(ppl_list)][['NACCID'] + Ccol]
        
    abc_ppl = np.array(list(set(U['NACCID']).intersection(M['NACCID']).intersection(C['NACCID'])))
    ab_ppl = np.array(list(set(U['NACCID']).intersection(M['NACCID']) - set(abc_ppl)))
    ac_ppl = np.array(list(set(U['NACCID']).intersection(C['NACCID']) - set(abc_ppl)))
    a_ppl = np.array(list(set(U['NACCID']) - set(mri['NACCID']) - set(C['NACCID'])))
    assert(len(abc_ppl) + len(ab_ppl) + len(ac_ppl) + len(a_ppl) == len(U))
    
    abc = U[U['NACCID'].isin(abc_ppl)].merge(M[M['NACCID'].isin(abc_ppl)], on='NACCID')\
                                  .merge(C[C['NACCID'].isin(abc_ppl)][['NACCID'] + Ccol], on='NACCID').drop(['NACCID'], axis=1)
    ab = U[U['NACCID'].isin(ab_ppl)].merge(M[M['NACCID'].isin(ab_ppl)], on='NACCID').drop(['NACCID'], axis=1)
    ac = U[U['NACCID'].isin(ac_ppl)].merge(C[C['NACCID'].isin(ac_ppl)][['NACCID'] + Ccol] , on='NACCID').drop(['NACCID'], axis=1)
    a = U[U['NACCID'].isin(a_ppl)].drop(['NACCID'], axis=1)
    y = np.concatenate([abc[target].values, ab[target].values, ac[target].values, a[target].values])

    abc = torch.tensor(abc.drop(target, axis=1).to_numpy()).float().to(device)
    ab = torch.tensor(ab.drop(target, axis=1).to_numpy()).float().to(device)
    ac = torch.tensor(ac.drop(target, axis=1).to_numpy()).float().to(device)
    a = torch.tensor(a.drop(target, axis=1).to_numpy()).float().to(device)
    
    return abc, ab, ac, a, y, Utr, Mtr

Ucol, Mcol, Ccol = list(uds.drop(uds_drop_columns, axis=1).columns), \
               list(mri.drop(mri_drop_columns, axis=1).columns), \
               ['CSFABETA', 'CSFTTAU', 'CSFPTAU'] 

def obtain_inner_metrics(seed=1):
    abc, ab, ac, a, y, Utr, Mtr = prepare_autoencoder_data(inner_train_ppl)
    Hab = torch.concat([aPred(abc[:,:len(Ucol)]), bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)])], axis=1)
    Hac = torch.concat([aPred(abc[:,:len(Ucol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    Hbc = torch.concat([bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    ytr = y
    ytr_hat = abcPred.predict(torch.concat([abPred(Hab), acPred(Hac), bcPred(Hbc)], axis=1))

    abc, ab, ac, a, y, Utr, Mtr = prepare_autoencoder_data(inner_test_ppl)
    Hab = torch.concat([aPred(abc[:,:len(Ucol)]), bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)])], axis=1)
    Hac = torch.concat([aPred(abc[:,:len(Ucol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    Hbc = torch.concat([bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    yte = y
    yte_hat = abcPred.predict(torch.concat([abPred(Hab), acPred(Hac), bcPred(Hbc)], axis=1))
    return obtain_metrics(ytr, ytr_hat, yte, yte_hat, df_type='Hierachical', seed=seed)

def obtain_outer_metrics(seed=1):
    abc, ab, ac, a, y, Utr, Mtr = prepare_autoencoder_data(outer_train_ppl)
    Hab = torch.concat([aPred(abc[:,:len(Ucol)]), bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)])], axis=1)
    Hac = torch.concat([aPred(abc[:,:len(Ucol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    Hbc = torch.concat([bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    yabc = abcPred.predict(torch.concat([abPred(Hab), acPred(Hac), bcPred(Hbc)], axis=1))

    Hab = torch.concat([aPred(ab[:,:len(Ucol)]), bPred(ab[:,len(Ucol):])], axis=1)
    yab = abPred.predict(Hab)

    Hac = torch.concat([aPred(ac[:,:len(Ucol)]), cPred(ac[:,len(Ucol):])], axis=1)
    yac = acPred.predict(Hac)

    ya = aPred.predict(a)
    
    ytr = y
    ytr_hat = np.concatenate([yabc, yab, yac, ya], axis=0)
    
    abc, ab, ac, a, y, Utr, Mtr = prepare_autoencoder_data(outer_test_ppl)
    Hab = torch.concat([aPred(abc[:,:len(Ucol)]), bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)])], axis=1)
    Hac = torch.concat([aPred(abc[:,:len(Ucol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    Hbc = torch.concat([bPred(abc[:,len(Ucol):len(Ucol)+len(Mcol)]), cPred(abc[:,len(Ucol)+len(Mcol):])], axis=1)
    yabc = abcPred.predict(torch.concat([abPred(Hab), acPred(Hac), bcPred(Hbc)], axis=1))

    Hab = torch.concat([aPred(ab[:,:len(Ucol)]), bPred(ab[:,len(Ucol):])], axis=1)
    yab = abPred.predict(Hab)

    Hac = torch.concat([aPred(ac[:,:len(Ucol)]), cPred(ac[:,len(Ucol):])], axis=1)
    yac = acPred.predict(Hac)

    ya = aPred.predict(a)

    yte = y
    yte_hat = np.concatenate([yabc, yab, yac, ya], axis=0)
    return obtain_metrics(ytr, ytr_hat, yte, yte_hat, df_type='Hierachical', seed=seed)
    
    

master_inner = pd.DataFrame()
master_outer = pd.DataFrame()

for seed in range({STARTSEED}, {ENDSEED}):# Inner Join
    start = time.time()
    print("Random Seed {} Starting...".format(seed))

    inner_ppl = set(uds['NACCID']).intersection(mri['NACCID']).intersection(csf['NACCID'])
    inner_train_ppl, inner_test_ppl = train_test_split(np.array(list(inner_ppl)), test_size = 0.2, random_state=seed)
    outer_ppl = list(set(uds['NACCID'])- inner_ppl)
    outer_train_ppl, outer_test_ppl = train_test_split(np.array(outer_ppl), test_size = 0.2, random_state=seed)
    outer_train_ppl = np.concatenate([inner_train_ppl, outer_train_ppl])
    outer_test_ppl = np.concatenate([inner_test_ppl, outer_test_ppl])
    
    atr, ate = uds[uds['NACCID'].isin(outer_train_ppl)].reset_index(drop=True), uds[uds['NACCID'].isin(outer_test_ppl)].reset_index(drop=True)
    btr, bte = mri[mri['NACCID'].isin(outer_train_ppl)].reset_index(drop=True), mri[mri['NACCID'].isin(outer_test_ppl)].reset_index(drop=True)
    ctr, cte = csf[csf['NACCID'].isin(outer_train_ppl)].reset_index(drop=True), csf[csf['NACCID'].isin(outer_test_ppl)].reset_index(drop=True)

    (ab_atri, ab_btri), (ab_atei, ab_btei) = find_index(atr, btr), find_index(ate, bte)
    (ac_atri, ac_ctri), (ac_atei, ac_ctei) = find_index(atr, ctr), find_index(ate, cte)
    (bc_btri, bc_ctri), (bc_btei, bc_ctei) = find_index(btr, ctr), find_index(bte, cte)
    (abc_atri, abc_btri, abc_ctri), (abc_atei, abc_btei, abc_ctei) = find_all_index(atr, btr, ctr), find_all_index(ate, bte, cte)
    abc_abtri, abc_actri = find_index(atr.loc[ab_atri].reset_index(drop=True), atr.loc[ac_atri].reset_index(drop=True))
    abc_abtri, abc_bctri = find_index(atr.loc[ab_atri].reset_index(drop=True), btr.loc[bc_btri].reset_index(drop=True))
    abc_abtei, abc_actei = find_index(ate.loc[ab_atei].reset_index(drop=True), ate.loc[ac_atei].reset_index(drop=True))
    abc_abtei, abc_bctei = find_index(ate.loc[ab_atei].reset_index(drop=True), bte.loc[bc_btei].reset_index(drop=True))

    # Construct Tensor
    yatr = torch.tensor(atr['NACCAD3'].to_numpy()).long().to(device)
    yate = torch.tensor(ate['NACCAD3'].to_numpy()).long().to(device)
    ybtr = torch.tensor(btr.merge(atr, on='NACCID')['NACCAD3'].to_numpy()).long().to(device)
    ybte = torch.tensor(bte.merge(ate, on='NACCID')['NACCAD3'].to_numpy()).long().to(device)
    yctr = torch.tensor(ctr.merge(atr, on='NACCID')['NACCAD3'].to_numpy()).long().to(device)
    ycte = torch.tensor(cte.merge(ate, on='NACCID')['NACCAD3'].to_numpy()).long().to(device)

    atr = torch.tensor(atr.drop(uds_drop_columns, axis=1).to_numpy()).float().to(device)
    ate = torch.tensor(ate.drop(uds_drop_columns, axis=1).to_numpy()).float().to(device)
    btr = torch.tensor(btr.drop(mri_drop_columns, axis=1).to_numpy()).float().to(device)
    bte = torch.tensor(bte.drop(mri_drop_columns, axis=1).to_numpy()).float().to(device)
    ctr = torch.tensor(ctr.drop(csf_drop_columns, axis=1).to_numpy()).float().to(device)
    cte = torch.tensor(cte.drop(csf_drop_columns, axis=1).to_numpy()).float().to(device)

    aPred = PredModel([81, 32, 16, 3]).to(device)
    bPred = PredModel([155, 128, 64, 16, 3]).to(device)
    cPred = PredModel([3, 8, 4, 3]).to(device)
    abPred = PredModel([6, 4, 3]).to(device)
    acPred = PredModel([6, 4, 3]).to(device)
    bcPred = PredModel([6, 4, 3]).to(device)
    abcPred = PredModel([9, 4, 3]).to(device)
    
    aPred = train_pred(aPred, atr, yatr, ate, yate)
    bPred = train_pred(bPred, btr, ybtr, bte, ybte)
    cPred = train_pred(cPred, ctr, yctr, cte, ycte)

    Hatr, Hate = aPred(atr).detach(), aPred(ate).detach()
    Hbtr, Hbte = bPred(btr).detach(), bPred(bte).detach()
    Hctr, Hcte = cPred(ctr).detach(), cPred(cte).detach()
    Habtr, Habte = torch.concat([Hatr[ab_atri,], Hbtr[ab_btri,]], axis=1), torch.concat([Hate[ab_atei,], Hbte[ab_btei,]], axis=1)
    Hactr, Hacte = torch.concat([Hatr[ac_atri,], Hctr[ac_ctri,]], axis=1), torch.concat([Hate[ac_atei, ], Hcte[ac_ctei,]], axis=1)
    Hbctr, Hbcte = torch.concat([Hbtr[bc_btri,], Hctr[bc_ctri,]], axis=1), torch.concat([Hbte[bc_btei,], Hcte[bc_ctei,]], axis=1)
    abPred = train_pred(abPred, Habtr, yatr[ab_atri,], Habte, yate[ab_atei,])
    acPred = train_pred(acPred, Hactr, yatr[ac_atri,], Hacte, yate[ac_atei,])
    bcPred = train_pred(bcPred, Hbctr, ybtr[bc_btri,], Hbcte, ybte[bc_btei,])

    Habctr = torch.concat([abPred(Habtr.detach()).detach()[abc_abtri,], 
                           acPred(Hactr.detach()).detach()[abc_actri,], 
                           bcPred(Hbctr.detach()).detach()[abc_bctri,]], axis=1)
    Habcte = torch.concat([abPred(Habte.detach()).detach()[abc_abtei,], 
                           acPred(Hacte.detach()).detach()[abc_actei,], 
                           bcPred(Hbcte.detach()).detach()[abc_bctei,]], axis=1)
    abcPred = train_pred(abcPred, Habctr, yatr[ab_atri,][abc_abtri,], Habcte, yate[ab_atei,][abc_abtei,])


    inner_df = obtain_inner_metrics(seed)
    master_inner = pd.concat([master_inner, inner_df], axis=0)
    outer_df = obtain_outer_metrics(seed)
    master_outer = pd.concat([master_outer, outer_df], axis=0)
    pkl.dump([master_inner, master_outer], open('sim_50_{STARTSEED}_{ENDSEED}.pkl', 'wb'))
    print(master_inner[master_inner['Seed'] == seed])
    print(master_outer[master_outer['Seed'] == seed])
    end = time.time()
    print("One Iter Complete: Time-{:2f}s\n\n".format(end-start))










