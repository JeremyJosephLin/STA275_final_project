import pandas as pd
import numpy as np

################## Load UDS data  ##################
def correct_uds(uds):
    moca = pd.read_excel('../data/NACC Crosswalk Study Conversions.xlsx', sheet_name='MoCA2MMSE', na_values='NaN')
    mint = pd.read_excel('../data/NACC Crosswalk Study Conversions.xlsx', sheet_name='MINT2BNT')
    craft = pd.read_excel('../data/NACC Crosswalk Study Conversions.xlsx', sheet_name='Craft2Logic')
    numberFN = pd.read_excel('../data/NACC Crosswalk Study Conversions.xlsx', sheet_name='Number2DigitFN')
    numberFL = pd.read_excel('../data/NACC Crosswalk Study Conversions.xlsx', sheet_name='Number2DigitFL')
    numberBN = pd.read_excel('../data/NACC Crosswalk Study Conversions.xlsx', sheet_name='Number2DigitBN')
    numberBL = pd.read_excel('../data/NACC Crosswalk Study Conversions.xlsx', sheet_name='Number2DigitBL')
    
    conv_mmse = pd.DataFrame([moca.iloc[x,1] if x < 31 else np.nan for x in 
            uds.iloc[(uds['NACCMMSE'] < 0).values]['MOCATOTS'].values], 
            index=uds.iloc[(uds['NACCMMSE'] < 0).values].index, columns=['NACCMMSE'])
    uds['NACCMMSE'].mask(uds['NACCMMSE'] < 0, np.nan, inplace=True)
    uds = uds.fillna(conv_mmse)

    # MINT --> BNT 
    conv_bnt = pd.DataFrame([mint.iloc[x,1] if (x < 33 and x > -1) else np.nan for x in 
                uds.iloc[(uds['BOSTON'] < 0).values]['MINTTOTS'].values], 
                index=uds.iloc[(uds['BOSTON'] < 0).values].index, columns=['BOSTON'])
    uds['BOSTON'].mask(uds['BOSTON'] < 0, np.nan, inplace=True)
    uds = uds.fillna(conv_bnt)

    # Craft Story 21 --> Logical Memory
    conv_logic = pd.DataFrame([craft.iloc[x,1] if x < 26 else np.nan for x in 
                    uds.iloc[(uds['MEMUNITS'] < 0).values]['CRAFTDRE'].values], 
                    index=uds.iloc[(uds['MEMUNITS'] < 0).values].index, columns=['MEMUNITS'])
    uds['MEMUNITS'].mask(uds['MEMUNITS'] < 0, np.nan, inplace=True)
    uds = uds.fillna(conv_logic)

    # Number span, forward trials correct --> Digit span, number forward
    conv_numFN = pd.DataFrame([numberFN.iloc[x,1] if x < 15 else np.nan for x in 
                    uds.iloc[(uds['DIGIF'] < 0).values]['DIGFORCT'].values], 
                    index=uds.iloc[(uds['DIGIF'] < 0).values].index, columns=['DIGIF'])
    uds['DIGIF'].mask(uds['DIGIF'] < 0, np.nan, inplace=True)
    uds = uds.fillna(conv_numFN)

    # Number span, forward length --> Digit span, length forward
    conv_numFL = pd.DataFrame([numberFL.iloc[x-3,1] if x < 10 and x > 2 else (x if x == 0 else np.nan)
                    for x in uds.iloc[(uds['DIGIFLEN'] < 0).values]['DIGFORSL'].values], 
                    index=uds.iloc[(uds['DIGIFLEN'] < 0).values].index, columns=['DIGIFLEN'])
    uds['DIGIFLEN'].mask(uds['DIGIFLEN'] < 0, np.nan, inplace=True)
    uds = uds.fillna(conv_numFL)

    # Number span, backward trials correct --> Digit span, number backward
    conv_numBN = pd.DataFrame([numberBN.iloc[x,1] if x < 15 else np.nan for x in 
                    uds.iloc[(uds['DIGIB'] < 0).values]['DIGBACCT'].values], 
                    index=uds.iloc[(uds['DIGIB'] < 0).values].index, columns=['DIGIB'])
    uds['DIGIB'].mask(uds['DIGIB'] < 0, np.nan, inplace=True)
    uds = uds.fillna(conv_numBN)

    # Number span, backward length --> Digit span, length backward
    conv_numBL = pd.DataFrame([numberBL.iloc[x-1,1] if x < 9 and x > 1 else (x if x == 0 else np.nan)
                    for x in uds.iloc[(uds['DIGIBLEN'] < 0).values]['DIGBACLS'].values], 
                    index=uds.iloc[(uds['DIGIBLEN'] < 0).values].index, columns=['DIGIBLEN'])
    uds['DIGIBLEN'].mask(uds['DIGIBLEN'] < 0, np.nan, inplace=True)
    uds = uds.fillna(conv_numBL)
    return uds

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

def encode_target(uds):
    assert((uds[(uds['NACCUDSD'] == 1) & (uds['NACCALZP'] != 8)].shape[0] == 0) &
       (uds[(uds['NACCALZP'] == 8) & (uds['NACCUDSD'] != 1)].shape[0] == 0))
    uds['NACCAD3'] = uds[['NACCUDSD', 'NACCALZP']].apply(
        lambda x: "Healthy" if x['NACCUDSD'] == 1 else 
                  "MCI-AD" if (x['NACCUDSD'] == 3) & (x['NACCALZP'] in [1,2]) else
                  "Dementia-AD" if (x['NACCUDSD'] == 4) & (x['NACCALZP'] in [1,2]) else
                  np.nan, axis=1)
    uds['NACCAD5'] = uds[['NACCUDSD', 'NACCALZP']].apply(
        lambda x: "Healthy" if x['NACCUDSD'] == 1 else 
                  "MCI-AD" if (x['NACCUDSD'] == 3) & (x['NACCALZP'] in [1,2]) else
                  "MCI-NonAD" if (x['NACCUDSD'] == 3) & (x['NACCALZP'] in [3, 7]) else
                  "Dementia-AD" if (x['NACCUDSD'] == 4) & (x['NACCALZP'] in [1,2]) else
                  "Dementia-NonAD" if (x['NACCUDSD'] == 4) & (x['NACCALZP'] in [3,7]) else
                  np.nan, axis=1)
    return uds

def load_uds(file_path = "../data/investigator_uds_baseline.csv",
             dict_path = "../data/data_dictionary/uds_feature_dictionary_cleaned.csv"):
    uds = pd.read_csv(file_path)
    uds = correct_uds(uds)
    uds_dict = pd.read_csv(dict_path)
    uds = mask_uds_na(uds, uds_dict)
    uds['datetime'] = pd.to_datetime(uds['datetime'])
    uds = uds.sort_values(by=['datetime'])
    dropped = ['MOCATOTS','MINTTOTS','CRAFTDRE','DIGFORCT','DIGFORSL','DIGBACCT','DIGBACLS']
    uds = uds.drop(columns=dropped)
    uds = encode_target(uds)
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
    mri.rename(columns={'MRIYR': 'year', 'MRIMO': 'month', 'MRIDY': 'day'}, inplace=True)
    mri['datetime'] = pd.to_datetime(mri[['year','month','day']],yearfirst=True,infer_datetime_format=True)
    mri = mri.drop(['year','month','day'], axis=1)
    mri = mask_mri_na(mri)
    # Missing rate less than 10% (all 2873 patients are in UDS)
    mri = mri[(mri.isna().sum(axis=1) / 170) < 0.1]
    mri = mri.sort_values(by=['datetime'])
    return mri


################## Load CSF data  ##################

def load_csf(file_path = "../data/investigator_fcsf_nacc57.csv"):
    csf = pd.read_csv(file_path)
    csf.rename(columns={'CSFLPYR': 'year', 'CSFLPMO': 'month', 'CSFLPDY': 'day'}, inplace=True)
    csf['datetime'] = pd.to_datetime(csf[['year','month','day']],yearfirst=True,infer_datetime_format=True)
    csf = csf.drop(['month', 'day', 'year',
                    'CSFABMO', 'CSFABDY', 'CSFABYR',
                    'CSFPTMO', 'CSFPTDY', 'CSFPTYR', 
                    'CSFTTMO', 'CSFTTDY', 'CSFTTYR', 
                    'CSFTTMDX','CSFABMDX', 'CSFPTMDX'], axis=1)
    return csf


def load_feature_map(uds_path = "../data/data_dictionary/uds_feature_dictionary_cleaned.csv",
                     mri_path = "../data/data_dictionary/mri_feature_dictionary_cleaned.csv"):
    # Load UDS dataset feature dictionary
    uds_dict = pd.read_csv(uds_path)
    mri_dict = pd.read_csv(mri_path)
    return uds_dict, mri_dict