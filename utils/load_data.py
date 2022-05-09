import pandas as pd
import numpy as np

################## Load UDS data  ##################
def mask_uds_na(uds, uds_dict):
    uds['EDUC'] = uds['EDUC'].mask(uds['EDUC'] == 99, np.nan)
    for var in uds_dict[uds_dict['Category'] == 'CDF']['VariableName']:
        uds[var] = uds[var].mask(uds[var] == -4, np.nan)
    for var in uds_dict[uds_dict['Category'] == 'GDS']['VariableName']:
        if var != "NACCGDS":
            uds[var] = uds[var].mask(uds[var].isin([-4, 9]), np.nan)
        else:
            uds[var] = uds[var].mask(uds[var].isin([-4, 88]), np.nan)
    for var in uds_dict[uds_dict['Category'] == 'FAS']['VariableName']:
        uds[var] = uds[var].mask(uds[var].isin([8, 9, -4]), np.nan)
    for var in uds_dict[uds_dict['Category'] == 'NPI']['VariableName']:
        uds[var] = uds[var].mask(uds[var].isin([8, 9, -4]), np.nan)
    for var in uds_dict[uds_dict['Category'] == 'NEURO']['VariableName']:
        if var not in ['TRAILA', "TRAILB"]:
            uds[var] = uds[var].mask(uds[var].isin([88, 95, 96, 97, 98, 99, -4]), np.nan)
        else:
            uds[var] = uds[var].mask(uds[var].isin([995, 996, 997, 998, -4]), np.nan)
    return uds

def load_uds(file_path = "../data/investigator_uds_baseline.csv",
             dict_path = "../data/data_dictionary/uds_feature_dictionary_cleaned.csv"):
    uds = pd.read_csv(file_path)
    uds_dict = pd.read_csv(dict_path)
    uds = mask_uds_na(uds, uds_dict)
    return uds

################## Load MRI data  ##################

def mask_mri_na(df):
    # MRI ROIs missing values (see Handbook)
    for code in [8.8888, 88.8888, 888.8888, 8888.888, 8888.8888,
                 9.999, 99.9999, 999.9999, 9999.999, 9999.9999]:
        df = df.mask(df == code, np.nan)
    return df

def load_mri(file_path = '../data/investigator_mri_nacc57.csv'):
    mri_raw = pd.read_csv(file_path)
    mri_features_removed = ['NACCDICO','NACCNIFT','NACCMRFI','NACCNMRI','NACCMNUM','NACCMRSA','MRIMANU',
                        'MRIMODL','NACCMRIA', 'NACCMRDY', 'MRIT1', 'MRIT2', 'MRIDTI','MRIDWI', 'MRIFLAIR', 
                        'MRIOTHER', 'MRIFIELD', 'NACCMVOL', 'NACCADC']
    mri = mri_raw.loc[:, ~mri_raw.columns.isin(mri_features_removed)] 
    mri = mask_mri_na(mri)
    return mri


################## Load CSF data  ##################

def load_csf(file_path = "../data/investigator_fcsf_nacc57.csv"):
    csf = pd.read_csv(file_path)
    return csf


def load_feature_map(uds_path = "../data/data_dictionary/uds_feature_dictionary_cleaned.csv",
                     mri_path = "../data/data_dictionary/mri_feature_dictionary_cleaned.csv"):
    # Load UDS dataset feature dictionary
    uds_dict = pd.read_csv(uds_path)
    mri_dict = pd.read_csv(mri_path)
    return uds_dict, mri_dict