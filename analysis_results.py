
import os
import numpy as np
import pandas as pd

def summarize(csv_path, group_cols, metrics=("best_val_acc","test_acc","search_auc","params","flops")):
    df=pd.read_csv(csv_path)
    agg={}
    for m in metrics:
        if m in df.columns:
            agg[m]=["mean","std","median",lambda x: x.quantile(.75)-x.quantile(.25)]
    out=df.groupby(group_cols,dropna=False).agg(agg)
    out.columns=["_".join([a,("iqr" if b=="<lambda_0>" else b)]) for a,b in out.columns]
    return out.reset_index()

def pareto_front(df, performance="test_acc", cost="flops"):
    d=df[[performance,cost]].dropna().sort_values([cost,performance],ascending=[True,False])
    keep=[]; best=-np.inf
    for idx,row in d.iterrows():
        if row[performance]>best:
            keep.append(idx); best=row[performance]
    return df.loc[keep].sort_values(cost)

def friedman_test(csv_path, metric, block_cols=("dataset","budget","seed"), method_col="method"):
    from scipy.stats import friedmanchisquare
    df=pd.read_csv(csv_path)
    pivot=df.pivot_table(index=list(block_cols),columns=method_col,values=metric,aggfunc="mean").dropna()
    if pivot.shape[1] < 3 or len(pivot) < 2:
        return {"statistic":np.nan,"pvalue":np.nan,"n_blocks":len(pivot),"methods":list(pivot.columns)}
    stat,p=friedmanchisquare(*[pivot[c].values for c in pivot.columns])
    return {"statistic":float(stat),"pvalue":float(p),"n_blocks":len(pivot),"methods":list(pivot.columns)}

def wilcoxon_pair(csv_path, metric, method_a, method_b, block_cols=("dataset","budget","seed"), method_col="method"):
    from scipy.stats import wilcoxon
    df=pd.read_csv(csv_path)
    p=df[df[method_col].isin([method_a,method_b])].pivot_table(
        index=list(block_cols),columns=method_col,values=metric,aggfunc="mean").dropna()
    if method_a not in p or method_b not in p or len(p)==0:
        return {"statistic":np.nan,"pvalue":np.nan,"n":0}
    stat,pv=wilcoxon(p[method_a],p[method_b],zero_method="wilcox",alternative="two-sided")
    return {"statistic":float(stat),"pvalue":float(pv),"n":len(p)}

def cliffs_delta(x,y):
    x=np.asarray(x); y=np.asarray(y)
    gt=sum(a>b for a in x for b in y); lt=sum(a<b for a in x for b in y)
    return (gt-lt)/(len(x)*len(y)) if len(x)*len(y) else np.nan
