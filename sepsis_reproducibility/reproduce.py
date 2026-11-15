"""Reproduce the structured retrospective AmsterdamUMCdb analysis.

Input: an authorized, harmonized analytic matrix. Output: aggregate JSON only.
This module does not run an LLM, audit generated text, or provide clinical advice.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

LAC = ['lac0','lac6','lac12','lac24','lac48']
REQUIRED = ['admissionid','sofa0','sofa48','hr_sd','rr_sd',*LAC,
            'alert_red','mort28','vaso_escalation']

def bool_column(s):
    values = s.astype(str).str.lower().map({'true':True,'false':False,'1':True,'0':False})
    if values.isna().any():
        raise ValueError(f'Invalid or missing binary values in {s.name}')
    return values.astype(bool)

def load_matrix(path):
    df = pd.read_csv(path, usecols=lambda name: name in REQUIRED + ['agegroup','deterioration_ref'])
    missed = sorted(set(REQUIRED) - set(df))
    if missed:
        raise ValueError(f'Missing required columns: {", ".join(missed)}')
    if df.empty or df.admissionid.isna().any() or df.admissionid.duplicated().any():
        raise ValueError('The matrix must have one row per nonmissing admission key')
    for col in ['sofa0','sofa48','hr_sd','rr_sd',*LAC]:
        df[col] = pd.to_numeric(df[col], errors='raise').replace([np.inf,-np.inf],np.nan)
    for col in ['alert_red','mort28','vaso_escalation','deterioration_ref']:
        if col in df:
            df[col] = bool_column(df[col])
    composite = df.mort28 | df.vaso_escalation
    if 'deterioration_ref' in df and not df.deterioration_ref.equals(composite):
        raise ValueError('Composite must be mortality or vasopressor escalation')
    df['deterioration_ref'] = composite
    return df

def rules(df):
    """Replay legacy rules on measured values. No middle-window imputation is used."""
    delta = df.sofa48 - df.sofa0
    clearance = (df.lac0 - df.lac48) / (df.lac0.where(df.lac0 > 0) + 1e-5)
    return pd.DataFrame({
        'sofa_increase': delta >= 2,
        'lactate_clearance': (df.lac48 > 2) & (clearance < .10),
        'heart_rate_sd_and_sofa': (df.hr_sd < 10) & (delta > 0)
    }, index=df.index)

DAY_MS = 86_400_000

def with_admission_outcomes(df, admissions):
    cols = ['patientid','admissionid','admittedat','dateofdeath']
    missing = set(cols) - set(admissions)
    if missing:
        raise ValueError('Missing admissions columns: ' + ', '.join(sorted(missing)))
    d = df.merge(admissions[cols], on='admissionid', how='left', validate='one_to_one', sort=False)
    if d[['patientid','admittedat']].isna().any().any():
        raise ValueError('All analytic admissions must link to a patient and admission time')
    admitted = pd.to_numeric(d.admittedat,errors='raise')
    death = pd.to_numeric(d.dateofdeath,errors='raise')
    d['legacy_mort28'] = d.mort28
    d['mort28'] = (death-admitted).between(0,28*DAY_MS)
    d['deterioration_ref'] = d.mort28 | d.vaso_escalation
    return d

def ratio(x, red, patient=None):
    n1, n0 = int(red.sum()), int((~red).sum())
    x1, x0 = int((red & x).sum()), int((~red & x).sum())
    d = {'red_events':x1,'red_total':n1,'no_red_events':x0,'no_red_total':n0}
    if min(x1, x0, n1, n0) == 0:
        return {**d, 'risk_ratio':None, '95_ci':None}
    rr = (x1/n1)/(x0/n0)
    if patient is None:
        se = math.sqrt(1/x1-1/n1+1/x0-1/n0)
    else:
        # Sandwich variance for log(p_red) - log(p_no_red), grouped by patient.
        influence = np.where(red, (x.astype(float)-x1/n1)/x1, -(x.astype(float)-x0/n0)/x0)
        score = pd.Series(influence,index=x.index).groupby(patient).sum()
        g = len(score)
        if g < 2:
            raise ValueError('At least two patient clusters are required')
        se = math.sqrt(float((score**2).sum())*g/(g-1))
    return {**d, 'risk_ratio':rr, '95_ci':[rr*math.exp(-1.96*se),rr*math.exp(1.96*se)]}

def metrics(obs, pred):
    return {'R2':float(r2_score(obs,pred)),'MSE':float(mean_squared_error(obs,pred)),
            'MAE':float(mean_absolute_error(obs,pred))}

def summarize(df):
    flags = rules(df)
    red = flags.any(axis=1)
    m6, m12 = df.lac6.isna(), df.lac12.isna()
    miss = m6 | m12
    later = df.lac24.notna() | df.lac48.notna()
    all_bracketed = miss & (~m6 | (df.lac0.notna() & (df.lac12.notna() | later))) & (~m12 | ((df.lac0.notna() | df.lac6.notna()) & later))
    c = df.dropna(subset=LAC+['sofa0','sofa48','hr_sd','rr_sd']).query('lac0 >= 0.3')
    benchmark = {'n_encounters':len(c)}
    if len(c)>=2:
        v1 = .85*c.lac0 + .20*(c.sofa48-c.sofa0) + .05*(10-c.hr_sd) + .03*c.rr_sd
        diff = c.lac24 - c.lac0
        y = np.r_[c.lac6,c.lac12]
        benchmark.update({'n_values':2*len(c),'frozen_formula':metrics(y,np.r_[v1,v1]),
            'interpolation_0h_24h':metrics(y,np.r_[c.lac0+.25*diff,c.lac0+.50*diff])})
    pats={}
    labels=flags.apply(lambda row: '+'.join(row.index[row]) or 'none', axis=1)
    for pat in sorted(labels.unique()):
        take=labels.eq(pat)
        pats[pat]={'n':int(take.sum()), 'mortality_rate':float(df.mort28[take].mean()),
                   'composite_rate':float(df.deterioration_ref[take].mean())}
    return {'n_encounters':len(df),'red_alerts':int(red.sum()),'exact_alert_matches':int((red==df.alert_red).sum()),
            'outcomes':{col:ratio(df[col],red,df.get('patientid')) for col in ['mort28','deterioration_ref','vaso_escalation']},
            'missing_lactate':{'either':int(miss.sum()),'both':int((m6&m12).sum()),
                               'later_24h_48h':int((miss&later).sum()),
                               'all_missing_values_bracketed':int(all_bracketed.sum()),
                               'red_alerts':int((red&miss).sum())},
            'complete_case_benchmark':benchmark,'rule_patterns':pats}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix',type=Path,required=True)
    parser.add_argument('--admissions',type=Path,required=True,help='authorized AmsterdamUMCdb admissions.csv')
    parser.add_argument('--output',type=Path,required=True,help='path to a new aggregate JSON')
    args=parser.parse_args()
    if args.output.suffix.lower()!='.json' or args.output.resolve() in {args.matrix.resolve(),args.admissions.resolve()}:
        parser.error('Output must be a different .json file')
    matrix=load_matrix(args.matrix)
    source=pd.read_csv(args.admissions,usecols=['patientid','admissionid','admittedat','dateofdeath'])
    corrected=with_admission_outcomes(matrix,source)
    result=summarize(corrected)
    result.update(n_patients=int(corrected.patientid.nunique()),
                  legacy_mortality_events=int(corrected.legacy_mort28.sum()),
                  corrected_mortality_events=int(corrected.mort28.sum()),
                  mortality_flags_changed=int((corrected.mort28!=corrected.legacy_mort28).sum()))
    with args.matrix.open('rb') as f:
        result['input_sha256']=hashlib.file_digest(f,'sha256').hexdigest()
    with args.output.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False); f.write('\n')
if __name__=='__main__':
    main()
