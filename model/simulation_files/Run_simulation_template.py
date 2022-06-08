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


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
scaler = StandardScaler()
pca = None

def pca_transform(df, var_dict, n_component, pca_thres = 0.8):
    global pca
    pca = PCA(n_components=n_component)
    to_ret = pd.DataFrame()
    for cat in var_dict['Category'].unique():
        if cat != 'DEMO':
            var_names = var_dict[var_dict['Category'] == cat]['VariableName'].values
            var_names = set(var_names).intersection(set(df.columns))
            if len(var_names) > n_component:
                pca_transformed = pca.fit_transform(scaler.fit_transform(df.loc[:,var_names]))
                num_selected_1 = np.sum(pca.explained_variance_ratio_ > pca_thres)
                num_selected_2 = np.argmax(-np.diff(pca.explained_variance_ratio_, n=1) > 0.1) + 1
                num_selected = max(num_selected_1, num_selected_2)
                temp = pd.DataFrame(pca_transformed[:,:num_selected])
                temp.columns = ["{}_{}".format(cat, i+1) for i in range(num_selected)]
                print(cat, pca.explained_variance_ratio_, num_selected)
                to_ret = pd.concat([to_ret, temp], axis=1)
    return to_ret


uds_pca = pca_transform(uds.drop(uds_drop_columns, axis=1), uds_dict, 5, pca_thres=0.2)
uds_pca = pd.concat([uds[['NACCID', 'SEX', 'NACCAGE', 'EDUC', 'NACCAPOE', target]].reset_index(drop=True), uds_pca], axis=1)

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

kmeans = KMeans(n_clusters=3, random_state=0).fit(uds_pca.drop(['NACCID', 'NACCAD3'], axis=1))
uds_pca['label'] = kmeans.labels_


mri_pca = pca_transform(mri.drop(mri_drop_columns, axis=1), mri_dict, 5, pca_thres=0.1)
mri_pca['NACCID'] = mri.reset_index()['NACCID']
mri_pca = mri_pca[mri_pca['NACCID'].isin(uds_pca['NACCID'])]
uds_mri_merged = uds_pca.merge(mri_pca, on='NACCID', how='inner')

mri_group = mri_pca.merge(uds_pca[['NACCID', 'label']], on='NACCID', how='inner')
mri_info = mri_group.drop(['NACCID'], axis=1).groupby('label').mean().reset_index()
uds_mri_all = uds_pca[~uds_pca['NACCID'].isin(uds_mri_merged['NACCID'])]
uds_mri_all = uds_mri_all.merge(mri_info, on='label')
uds_mri_all = pd.concat([uds_mri_all, uds_mri_merged], axis=0)

csf = csf[csf['NACCID'].isin(uds_pca['NACCID'])]
csf_group = csf.drop(['CSFABMD', 'CSFTTMD', 'CSFPTMD'], axis=1).merge(uds_pca[['NACCID', 'label']], how='inner')
csf_info = csf_group.drop('NACCID', axis=1).groupby('label').mean().reset_index()
uds_mri_csf_merged = uds_mri_all.merge(csf_group.drop('label', axis=1), on='NACCID', how='inner')
uds_mri_csf_all = uds_mri_all[~uds_mri_all['NACCID'].isin(csf_group['NACCID'])]
uds_mri_csf_all = uds_mri_csf_all.merge(csf_info, on='label', how='inner')
uds_mri_csf_all = pd.concat([uds_mri_csf_all, uds_mri_csf_merged], axis=0)

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
    metrics_df = obtain_metrics(ytr, clf.predict(Xtr), yte, clf.predict(Xte), confusion_metrix=confusion_metrix, df_type = df_type, seed=seed)
    return metrics_df

def train_logistic(Xtr, ytr, C=0.4, l1_ratio=0.3):
    clf = LogisticRegression(random_state=48, max_iter=1500, solver='saga', penalty='elasticnet', C=C, l1_ratio=l1_ratio)
    clf.fit(Xtr, ytr)
    return clf 

class AutoEncoder(torch.nn.Module):
    def __init__(self, e_dim, d_dim, seed=48):
        super(AutoEncoder, self).__init__()
        seed_torch(seed)
        e_list = [nn.Linear(e_dim[0], e_dim[1])]
        for i in range(2, len(e_dim)):
            e_list.append(nn.ReLU())
#             e_list.append(nn.Dropout(0.1))
            e_list.append(nn.Linear(e_dim[i-1], e_dim[i]))
        d_list = [nn.Linear(d_dim[0], d_dim[1])]
        for i in range(2, len(d_dim)):
            d_list.append(nn.ReLU())
#             d_list.append(nn.Dropout(0.1))
            d_list.append(nn.Linear(d_dim[i-1], d_dim[i]))
        self.encoder_layers = nn.Sequential(*e_list)
        self.decoder_layers = nn.Sequential(*d_list)
    def encode(self, X):
        return self.encoder_layers(X)
    def decode(self, Z):
        return self.decoder_layers(Z)
    def loss(self, X1, X2):
        Z = self.encoder_layers(X1)
        X_hat = self.decoder_layers(Z)
        return torch.mean((X2-X_hat)**2)
    
def prepare_autoencoder_data(ppl_list, original=True, U_scaler=None, M_scaler=None, C_scaler=None):
    if original:
        U = uds[uds['NACCID'].isin(ppl_list)][['NACCID', 'NACCAD3'] + Ucol]
    else:
        U = uds_pca[uds_pca['NACCID'].isin(ppl_list)].drop('label', axis=1)
    if U_scaler is None:
        U_scaler = StandardScaler()
        U[Ucol] = U_scaler.fit_transform(U[Ucol])
    else:
        U[Ucol] = U_scaler.transform(U[Ucol])
    Utr = torch.tensor(U[Ucol].to_numpy()).float().to(device)
 
    if original:
        M = mri[mri['NACCID'].isin(ppl_list)][['NACCID'] + Mcol]
    else:
        M = mri_pca[mri_pca['NACCID'].isin(ppl_list)]
    if M_scaler is None:
        M_scaler = StandardScaler()
        M[Mcol] = M_scaler.fit_transform(M[Mcol])
    else:
         M[Mcol] = M_scaler.transform(M[Mcol])
    Mtr = torch.tensor(M[Mcol].to_numpy()).float().to(device)

    C = csf[csf['NACCID'].isin(ppl_list)][['NACCID'] + Ccol]
    if C_scaler is None:
        C_scaler = StandardScaler()
        C[Ccol] = C_scaler.fit_transform(C[Ccol])
    else:
        C[Ccol] = C_scaler.transform(C[Ccol])
        
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
    y = torch.tensor(np.concatenate([abc[target].values, ab[target].values, ac[target].values, a[target].values])).float().to(device)

    abc = torch.tensor(abc.drop(target, axis=1).to_numpy()).float().to(device)
    ab = torch.tensor(ab.drop(target, axis=1).to_numpy()).float().to(device)
    ac = torch.tensor(ac.drop(target, axis=1).to_numpy()).float().to(device)
    a = torch.tensor(a.drop(target, axis=1).to_numpy()).float().to(device)
    
    return abc, ab, ac, a, y, Utr, Mtr, U_scaler, M_scaler, C_scaler


def train_Uenc(U_enc, U_dec, Utr, display_intvl=50):
    U_enc = AutoEncoder(U_enc, U_dec).to(device)
    optimizer = torch.optim.Adam(U_enc.parameters(), lr=1e-4, weight_decay=1e-5)
    U_enc_loss = U_enc.loss(Utr, Utr)
    last_loss, loss = 0, 0
    for epoch in range(5000):
        loss =  U_enc.loss(Utr, Utr)
        loss.backward()
        optimizer.step()
        if (epoch % display_intvl == 0 or torch.abs(last_loss - loss) < 1e-9):
            print('U-Enc Epoch {}:  Loss: {:.3f}'.format(epoch, loss.item()))
        if torch.abs(last_loss - loss) < 1e-9:
            break
        last_loss = loss
    return U_enc

def train_Menc(M_enc, M_dec, Mtr, display_intvl=50):
    M_enc = AutoEncoder(M_enc, M_dec).to(device)
    optimizer = torch.optim.Adam(M_enc.parameters(), lr=1e-4, weight_decay=1e-5)
    last_loss, loss = 0, 0
    for epoch in range(5000):
        loss =  M_enc.loss(Mtr, Mtr)
        loss.backward()
        optimizer.step()
        if (epoch % display_intvl == 0 or torch.abs(last_loss - loss) < 1e-9):
            print('M-Enc Epoch {}:  Loss: {:.3f}'.format(epoch, loss.item()))
        if torch.abs(last_loss - loss) < 1e-9:
            break
        last_loss = loss
    return M_enc

def train_UMenc(UM_enc, UM_dec, ab, display_intvl=50):
    UM_enc = AutoEncoder(UM_enc, UM_dec).to(device)
    optimizer = torch.optim.Adam(UM_enc.parameters(), lr=1e-3, weight_decay=1e-5)
    last_loss, loss = 0, 0
    for epoch in range(5000):
        UM_enc_loss = UM_enc.loss(ab[:,:len(Ucol)], ab[:,len(Ucol):]) 
        UM_overlap_loss = torch.mean((UM_enc.encode(ab[:,:len(Ucol)]) - M_enc.encode(ab[:,len(Ucol):]))**2)
        loss =  UM_enc_loss + UM_overlap_loss
        loss.backward()
        optimizer.step()
        if (epoch % display_intvl == 0 or torch.abs(last_loss - loss) < 1e-9):
            print('UM-ENC Epoch {}:  Loss: {:.3f}'.format(epoch, loss.item()))
        if torch.abs(last_loss - loss) < 1e-9:
            break
        last_loss = loss
    return UM_enc

def train_UCenc(enc_dim, dec_dim, ac, display_intvl=50):
    UC_enc = AutoEncoder(enc_dim, dec_dim).to(device)
    optimizer = torch.optim.Adam(UC_enc.parameters(), lr=1e-4, weight_decay=1e-5)
    last_loss, loss = 0, 0
    for epoch in range(5000):
        loss = UC_enc.loss(ac[:,:len(Ucol)], ac[:,len(Ucol):]) # Project M to M, autoencoder loss
        loss.backward()
        optimizer.step()
        if (epoch % display_intvl == 0 or torch.abs(last_loss - loss) < 1e-9):
            print('UC-Enc Epoch {}:  Loss: {:.3f}'.format(
                epoch, loss.item()))
        if torch.abs(last_loss - loss) < 1e-9:
            break
        last_loss = loss
    return UC_enc

def construct_latent(abc, ab, ac, a):
    Habc = torch.concat([U_enc.encode(abc[:,:len(Ucol)]), 
                         M_enc.encode(abc[:,len(Ucol):len(Ucol) + len(Mcol)]),
                         abc[:,len(Ucol) + len(Mcol):]], axis=1)
    Hab = torch.concat([U_enc.encode(ab[:,:len(Ucol)]), 
                        M_enc.encode(ab[:,len(Ucol):len(Ucol) + len(Mcol)]),
                        UC_enc.decode(UC_enc.encode(ab[:,:len(Ucol)]))], axis=1)
    Hac = torch.concat([U_enc.encode(ac[:,:len(Ucol)]), 
                        UM_enc.encode(ac[:,:len(Ucol)]),
                        ac[:,len(Ucol):]], axis=1)
    Ha = torch.concat([U_enc.encode(a[:,:len(Ucol)]), 
                       UM_enc.encode(a[:,:len(Ucol)]),
                       UC_enc.decode(UC_enc.encode(a[:,:len(Ucol)]))], axis=1)
    H_all = torch.concat([Habc, Hab, Hac, Ha], axis=0)
    return H_all


def train_test_split_modality(df, innerTr, innerTe, outerTr, outerTe):
    dfTr = pd.concat([df[df['NACCID'].isin(innerTr)].sort_values('NACCID').reset_index(drop=True), 
                      df[df['NACCID'].isin(set(outerTr) - set(innerTr))].reset_index(drop=True)], axis=0).reset_index(drop=True)
    dfTe = pd.concat([df[df['NACCID'].isin(innerTe)].sort_values('NACCID').reset_index(drop=True), 
                      df[df['NACCID'].isin(set(outerTe) - set(innerTe))].reset_index(drop=True)], axis=0).reset_index(drop=True)
    return dfTr, dfTe

mri, csf = mri[mri['NACCID'].isin(uds['NACCID'])], csf[csf['NACCID'].isin(uds['NACCID'])]
uds, mri, csf = uds.sort_values('NACCID'), mri.sort_values('NACCID'), csf.sort_values('NACCID')

def prepare_X_Y(udsT, mriT, csfT, scalerU=None, scalerM=None, scalerC=None):
    XU, XM, XC = udsT.drop(uds_drop_columns, axis=1), mriT.drop(mri_drop_columns, axis=1), csfT.drop(csf_drop_columns, axis=1)
    if scalerU is None:
        scalerU, scalerM, scalerC = StandardScaler(), StandardScaler(), StandardScaler()
        XU, XM, XC = scalerU.fit_transform(XU).T, scalerM.fit_transform(XM).T, scalerC.fit_transform(XC).T
    else:
        XU, XM, XC = scalerU.transform(XU).T, scalerM.transform(XM).T, scalerC.transform(XC).T
    XU, XM, XC = torch.tensor(XU).float(), torch.tensor(XM).float(), torch.tensor(XC).float()

    YU, YM, YC = udsT['NACCAD3'], mriT.merge(uds, on='NACCID')['NACCAD3'], csfT.merge(uds, on='NACCID')['NACCAD3']
    YU, YM, YC = onehot_encoder.transform(YU.values.reshape(-1, 1)),onehot_encoder.transform(YM.values.reshape(-1, 1)),onehot_encoder.transform(YC.values.reshape(-1, 1))
    YU, YM, YC = torch.tensor(YU.T).float(), torch.tensor(YM.T).float(), torch.tensor(YC.T).float()
    return XU.to(device), XM.to(device), XC.to(device), YU.to(device), YM.to(device), YC.to(device), scalerU, scalerM, scalerC

def train_weights(W, X, H, E, Q, mu, verbose = False):
    for i in range(100):
        G = torch.diag((W**2).sum(1))
        W_new = torch.inverse(X.matmul(X.T) + (2*beta) / mu * G).matmul(X).matmul((H + E - Q / mu).T)
        if ((W_new - W)**2).sum() < 1e-6:
            break
        W = W_new
    if verbose and i == 100:
        print("Not converged")
    return W

def update_E(E, W, X, H, Q, mu):
    E_new = torch.zeros_like(E)
    temp = W.T.matmul(X) - H + Q / mu
    E_new = E_new + (temp > gamma / mu).int() * temp - gamma/mu
    E_new = E_new + (temp < -gamma / mu).int() * temp + gamma/mu
    return E_new

def calculate_accuracy(XU, XM, XC, YU, YM, YC, WU, WM, WC, P, nH):
    HU, HM, HC = WU.T.matmul(XU), WM.T.matmul(XM), WC.T.matmul(XC)
    H = (HU[:,:nH] + HM[:,:nH] + HC[:,:nH]) / 3
    YU_hat = P.matmul(torch.concat([H, HU[:,nH:]], axis=1)).argmax(axis=0)
    YM_hat = P.matmul(torch.concat([H, HM[:,nH:]], axis=1)).argmax(axis=0)
    YC_hat = P.matmul(torch.concat([H, HC[:,nH:]], axis=1)).argmax(axis=0)
    return "U-acc: {:.2f}%\tM-acc: {:.2f}%\tC-acc: {:.2f}%".format(
        (YU_hat == YU.argmax(0)).float().mean() * 100,
        (YM_hat == YM.argmax(0)).float().mean() * 100,
        (YC_hat == YC.argmax(0)).float().mean() * 100)

def predict_inner(XU, XM, XC, Y, WU, WM, WC, P, nH):
    HU, HM, HC = WU.T.matmul(XU), WM.T.matmul(XM), WC.T.matmul(XC)
    H = (HU[:,:nH] + HM[:,:nH] + HC[:,:nH]) / 3
    Y_hat =  P.matmul(H).argmax(axis=0)
    return Y_hat

def predict_outer(abc, ab, ac, a, y, WU, WM, WC, P):
    Yabc = P.matmul((WU.T.matmul(abc[:,:XU.shape[0]].T) + \
                         WM.T.matmul(abc[:,XU.shape[0]:XU.shape[0] + XM.shape[0]].T) + \
                         WC.T.matmul(abc[:,XU.shape[0] + XM.shape[0]:].T))/3).argmax(0)
    Yab = P.matmul((WU.T.matmul(ab[:,:XU.shape[0]].T) + WM.T.matmul(ab[:,XU.shape[0]:].T))/2).argmax(0)
    Yac = P.matmul((WU.T.matmul(ac[:,:XU.shape[0]].T) + WC.T.matmul(ac[:,XU.shape[0]:].T))/2).argmax(0)
    Ya = P.matmul(WU.T.matmul(a.T)).argmax(0)
    Y_hat = torch.concat([Yabc, Yab, Yac, Ya], axis=0)
    return Y_hat  

# maxiter, stop, rho, max_mu =1000, 1, 1.5, 1e6
def train_latent_features(maxiter=1000, stop = 1, rho = 1.5, mu=1e-2, max_mu = 1e6, display_intvl=20, seed=1):
#     Train individual weights
    seed_torch(seed)
    H = torch.randn(kH, nH).to(device)
    P = torch.randn(3, kH).to(device)
    WU, WM, WC = torch.randn(kU, kH).to(device), torch.randn(kM, kH).to(device), torch.randn(kC, kH).to(device)
    HU, EU, QU = torch.randn(kH, XU.shape[1]-nH).to(device), torch.randn(kH, XU.shape[1]).to(device), torch.ones(kH, XU.shape[1]).to(device)
    HM, EM, QM = torch.randn(kH, XM.shape[1]-nH).to(device), torch.randn(kH, XM.shape[1]).to(device), torch.ones(kH, XM.shape[1]).to(device)
    HC, EC, QC = torch.randn(kH, XC.shape[1]-nH).to(device), torch.randn(kH, XC.shape[1]).to(device), torch.ones(kH, XC.shape[1]).to(device)
    for i in range(maxiter):
        WU = train_weights(WU, XU, torch.concat([H, HU], axis=1), EU, QU, mu)
        WM = train_weights(WM, XM, torch.concat([H, HM], axis=1), EM, QM, mu)
        WC = train_weights(WC, XC, torch.concat([H, HC], axis=1), EC, QC, mu)

        EU_new = update_E(EU, WU, XU, torch.concat([H, HU], axis=1), QU, mu)
        EM_new = update_E(EM, WM, XM, torch.concat([H, HM], axis=1), QM, mu)
        EC_new = update_E(EC, WC, XC, torch.concat([H, HC], axis=1), QC, mu)

        HU = torch.inverse((P.T.matmul(P)) + mu * torch.eye(kH).to(device)).matmul(
            P.T.matmul(YU[:,nH:]) + mu * (WU.T.matmul(XU[:,nH:]) - EU[:,nH:] + QU[:,nH:] / mu))
        HM = torch.inverse((P.T.matmul(P)) + mu * torch.eye(kH).to(device)).matmul(
            P.T.matmul(YM[:,nH:]) + mu * (WM.T.matmul(XM[:,nH:]) - EM[:,nH:] + QM[:,nH:] / mu))
        HC = torch.inverse((P.T.matmul(P)) + mu * torch.eye(kH).to(device)).matmul(
            P.T.matmul(YC[:,nH:]) + mu * (WC.T.matmul(XC[:,nH:]) - EC[:,nH:] + QC[:,nH:] / mu))

        # train global parameters
        H = torch.inverse(P.T.matmul(P) + mu * torch.eye(kH).to(device) * 3).matmul(P.T.matmul(YU[:,:nH]) + mu * (
        (WU.T.matmul(XU[:,:nH]) - EU[:,:nH] + QU[:,:nH] / mu) + 
        (WM.T.matmul(XM[:,:nH]) - EM[:,:nH] + QM[:,:nH] / mu) + 
        (WC.T.matmul(XC[:,:nH]) - EC[:,:nH] + QC[:,:nH] / mu)
        ))

        H_all = torch.concat([H, HU, HM, HC], axis=1)
        Y_all = torch.concat([YU[:,:nH], YU[:,nH:], YM[:,nH:], YC[:,nH:]],axis=1)
        P = Y_all.matmul(H_all.T).matmul(torch.inverse(H_all.matmul(H_all.T) + eta * torch.eye(kH).to(device)))

        QU = QU + mu * (WU.T.matmul(XU) - torch.concat([H, HU], axis=1) - EU)
        QM = QM + mu * (WM.T.matmul(XM) - torch.concat([H, HM], axis=1) - EM)
        QC = QC + mu * (WC.T.matmul(XC) - torch.concat([H, HC], axis=1) - EC)

        mu = min(rho * mu, max_mu)

        errU = torch.abs(WU.T.matmul(XU) - torch.concat([H, HU], axis=1) - EU).max()
        errM = torch.abs(WM.T.matmul(XM) - torch.concat([H, HM], axis=1) - EM).max()
        errC = torch.abs(WC.T.matmul(XC) - torch.concat([H, HC], axis=1) - EC).max()
        error = max(errU, errM, errC)
        if error < stop:
            break
        if i == 0 or (i+1) % display_intvl == 0:
            inner_Y_hat = predict_inner(XU, XM, XC, YU[:,:len(inner_train_ppl)], WU, WM, WC, P, len(inner_train_ppl))
            abc, ab, ac, a, y, _, _, _, _, _ = prepare_autoencoder_data(outer_train_ppl, True, U_scaler, M_scaler, C_scaler)
            outer_Y_hat = predict_outer(abc, ab, ac, a, y, WU, WM, WC, P)
            print("Iter {}:  error-{:.3f}\t{}\tInner: {:.3f}%\tOuter: {:.3f}% ".format(i+1, error, 
                                                    calculate_accuracy(XU, XM, XC, YU, YM, YC, WU, WM, WC, P, len(inner_train_ppl)),
                                                    ((inner_Y_hat == YU[:,:len(inner_train_ppl)].argmax(0)).float().mean()).item()*100,
                                                    (outer_Y_hat == y).detach().cpu().numpy().mean() *100))
    if i == maxiter:
        print("Not Converged")
    return WU, WM, WC, P

master_inner = pd.DataFrame()
master_outer = pd.DataFrame()

for seed in range({STARTSEED}, {ENDSEED}):# Inner Join
    start = time.time()
    print("Random Seed {} Starting...".format(seed))
    inner_ppl = set(uds['NACCID']).intersection(mri['NACCID']).intersection(csf['NACCID'])
    inner_train_ppl, inner_test_ppl = train_test_split(np.array(list(inner_ppl)), test_size = 0.2, random_state=seed)
    inner_train = uds_mri_csf_all[uds_mri_csf_all['NACCID'].isin(inner_train_ppl)]
    inner_test = uds_mri_csf_all[uds_mri_csf_all['NACCID'].isin(inner_test_ppl)]
    Xi_train, yi_train = inner_train.drop(['NACCID', target], axis=1), inner_train[target]
    Xi_test, yi_test = inner_test.drop(['NACCID', 'NACCAD3'], axis=1), inner_test['NACCAD3']

    clf_inner = train_logistic(Xi_train, yi_train)
    inner_df = print_summary_clf(clf_inner, Xi_train, yi_train, Xi_test, yi_test, df_type = 'inner', seed=seed)
    master_inner = pd.concat([master_inner, inner_df], axis=0)
    
    # Group Info
    outer_ppl = list(set(uds_mri_csf_all['NACCID'])- inner_ppl)
    outer_train_ppl, outer_test_ppl = train_test_split(np.array(outer_ppl), test_size = 0.2, random_state=seed)
    outer_train_ppl = np.concatenate([outer_train_ppl, inner_train_ppl])
    outer_test_ppl = np.concatenate([outer_test_ppl, inner_test_ppl])
    outer_train = uds_mri_csf_all[uds_mri_csf_all['NACCID'].isin(outer_train_ppl)]
    outer_test = uds_mri_csf_all[uds_mri_csf_all['NACCID'].isin(outer_test_ppl)]
    Xo_train, yo_train = outer_train.drop(['NACCID', target], axis=1), outer_train[target]
    Xo_test, yo_test = outer_test.drop(['NACCID', target], axis=1), outer_test['NACCAD3']

    clf_group = train_logistic(Xo_train, yo_train)
    group_df = print_summary_clf(clf_group, Xo_train, yo_train, Xo_test, yo_test, df_type = 'group', seed=seed)
    master_outer = pd.concat([master_outer, group_df], axis=0)

    inner_df_outer = print_summary_clf(clf_inner, Xo_train, yo_train, Xo_test, yo_test, df_type = 'inner', seed=seed)
    master_outer = pd.concat([master_outer, inner_df_outer], axis=0)

    group_df_inner = print_summary_clf(clf_group, Xi_train, yi_train, Xi_test, yi_test, df_type = 'group', seed=seed)
    master_inner = pd.concat([master_inner, group_df_inner], axis=0)


    # AutoEncoder
    original = True

    if original:
        Ucol, Mcol, Ccol = list(uds.drop(uds_drop_columns, axis=1).columns), \
                       list(mri.drop(mri_drop_columns, axis=1).columns), \
                       ['CSFABETA', 'CSFTTAU', 'CSFPTAU'] 
    else:
        Ucol, Mcol, Ccol = list(uds_pca.drop(["NACCID", 'label', target], axis=1).columns), \
                       list(mri_pca.drop(['NACCID'], axis=1).columns), \
                       ['CSFABETA', 'CSFTTAU', 'CSFPTAU']

    abc, ab, ac, a, y, Utr, Mtr, U_scaler, M_scaler, C_scaler = prepare_autoencoder_data(outer_train_ppl, original)
    abcte, abte, acte, ate, yte, Ute, Mte, U_scaler, M_scaler, C_scaler = prepare_autoencoder_data(outer_test_ppl, original, U_scaler, M_scaler, C_scaler)
    U_enc = train_Uenc([81, 64, 16], [16, 32, 81], Utr, display_intvl=2500)
    M_enc = train_Menc([155, 64, 16, 8], [8, 16, 64, 155], Mtr, display_intvl=2500)
    UM_enc = train_UMenc([81, 32, 16, 8], [8, 16, 64, 155], ab, display_intvl=2500)
    UC_enc = train_UCenc([81, 32, 8, 3], [3, 3], ac, display_intvl=2500)

    H_all = construct_latent(abc, ab, ac, a)
    H_all_test = construct_latent(abcte, abte, acte, ate)
    clf_auto = train_logistic(H_all, y, C=0.4, l1_ratio=0.3)
    auto_outer = print_summary_clf(clf_auto, H_all, y, H_all_test, yte, df_type='auto', seed=seed)
    master_outer = pd.concat([master_outer, auto_outer], axis=0)

    XUi_train = inner_train[['NACCID']].merge(uds, on='NACCID')
    XUi_test = inner_test[['NACCID']].merge(uds, on='NACCID')
    XMi_train = inner_train[['NACCID']].merge(mri, on='NACCID')
    XMi_test = inner_test[['NACCID']].merge(mri, on='NACCID')
    XCi_train = inner_train[['NACCID']].merge(csf, on='NACCID')
    XCi_test = inner_test[['NACCID']].merge(csf, on='NACCID')

    Hi_train = torch.concat([U_enc.encode(torch.tensor(U_scaler.transform(XUi_train[Ucol])).float().to(device)),
    M_enc.encode(torch.tensor(M_scaler.transform(XMi_train[Mcol])).float().to(device)),
    torch.tensor(C_scaler.transform(XCi_train[Ccol])).float().to(device)], axis=1)

    Hi_test = torch.concat([U_enc.encode(torch.tensor(U_scaler.transform(XUi_test[Ucol])).float().to(device)),
    M_enc.encode(torch.tensor(M_scaler.transform(XMi_test[Mcol])).float().to(device)),
    torch.tensor(C_scaler.transform(XCi_test[Ccol])).float().to(device)], axis=1)

    auto_inner = print_summary_clf(clf_auto, Hi_train, yi_train, Hi_test, yi_test, df_type='auto', seed=seed)
    master_inner = pd.concat([master_inner, auto_inner], axis=0)
    
    # Latent Learning
    udsTr, udsTe = train_test_split_modality(uds, inner_train_ppl, inner_test_ppl, outer_train_ppl, outer_test_ppl)
    mriTr, mriTe = train_test_split_modality(mri, inner_train_ppl, inner_test_ppl, outer_train_ppl, outer_test_ppl)
    csfTr, csfTe = train_test_split_modality(csf, inner_train_ppl, inner_test_ppl, outer_train_ppl, outer_test_ppl)

    XU, XM, XC, YU, YM, YC, scalerU, scalerM, scalerC = prepare_X_Y(udsTr, mriTr, csfTr)
    XUTe, XMTe, XCTe, YUTe, YMTe, YCTe, scalerU, scalerM, scalerC = prepare_X_Y(udsTe, mriTe, csfTe, scalerU, scalerM, scalerC)

    kU, kM, kC = XU.shape[0], XM.shape[0], XC.shape[0]

    nH, kH = len(inner_train_ppl), 20
    lamb, beta, gamma, eta = 0.1, 0.1, 0.1, 0.1

    WU, WM, WC, P = train_latent_features(maxiter=40, stop = 1, rho = 1.5, mu=1e-2, max_mu = 1e6, seed=seed)
    
    inner_Ytr_hat = predict_inner(XU, XM, XC, YU[:,:len(inner_train_ppl)], WU, WM, WC, P, len(inner_train_ppl))
    inner_Yte_hat = predict_inner(XUTe, XMTe, XCTe, YUTe[:,:len(inner_test_ppl)], WU, WM, WC, P, len(inner_test_ppl))
    outer_Ytr_hat = predict_outer(abc, ab, ac, a, y, WU, WM, WC, P)
    outer_Yte_hat = predict_outer(abcte, abte, acte, ate, yte, WU, WM, WC, P)

    metrics_latent_inner = obtain_metrics(YU[:,:len(inner_train_ppl)].argmax(0), inner_Ytr_hat,
                                          YUTe[:,:len(inner_test_ppl)].argmax(0), inner_Yte_hat, df_type='Latent', seed=seed)
    master_inner = pd.concat([master_inner, metrics_latent_inner], axis=0)

    metrics_latent_outer = obtain_metrics(y, outer_Ytr_hat, yte, outer_Yte_hat, df_type='Latent', seed=seed)
    master_outer = pd.concat([master_outer, metrics_latent_outer], axis=0)
    
    print(master_inner[master_inner['Seed'] == seed])
    print(master_outer[master_outer['Seed'] == seed])
    pkl.dump([master_inner, master_outer], open('sim_50_{STARTSEED}_{ENDSEED}.pkl', 'wb'))
    end = time.time()
    print("One Iter Complete: Time-{:2f}s\n\n".format(end-start))


















