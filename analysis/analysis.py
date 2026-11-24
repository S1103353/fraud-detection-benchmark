import pandas as pd, numpy as np
import os
B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")+"/"
s12=pd.read_csv(B+"step12/step12_model_results.csv")
lr=pd.read_csv(B+"step03/step03_results.csv")
prev={'bank_fraud':0.0110,'creditcard':0.00167,'ecommerce':0.0221,'fraud_transactions':0.00579,'ieee':0.0350,'paysim':0.00129}
# variance decomposition PR-AUC across tree models
d=s12.copy()
gm=d.pr_auc.mean(); tot=((d.pr_auc-gm)**2).sum()
between=sum(len(g)*(g.pr_auc.mean()-gm)**2 for _,g in d.groupby('dataset'))
print("PR-AUC between-dataset share: %.3f"%(between/tot))
gm=d.f1.mean(); tot=((d.f1-gm)**2).sum(); between=sum(len(g)*(g.f1.mean()-gm)**2 for _,g in d.groupby('dataset'))
print("F1 between-dataset share: %.3f"%(between/tot))
# range within dataset
print(d.groupby('dataset').agg(prmin=('pr_auc','min'),prmax=('pr_auc','max'),f1min=('f1','min'),f1max=('f1','max')).round(3))
# projection
enr=pd.DataFrame([
 ['ieee','soft',0.770,0.692,0.172],['ieee','weighted',0.806,0.667,0.172],['ieee','stacking',0.620,0.820,0.172],
 ['bank_fraud','soft',0.319,0.599,0.055],['bank_fraud','weighted',0.389,0.505,0.055],['bank_fraud','stacking',0.211,0.809,0.055],
 ['fraud_transactions','soft',0.798,0.875,0.030],['fraud_transactions','weighted',0.839,0.859,0.030],['fraud_transactions','stacking',0.690,0.940,0.030],
 ['paysim','soft',0.929,0.997,0.033],['paysim','weighted',0.943,0.995,0.033],['paysim','stacking',0.917,0.998,0.033]],
 columns=['dataset','ens','p_s','r','pi_s'])
def proj(r):
    fpr=r.r*r.pi_s*(1-r.p_s)/(r.p_s*(1-r.pi_s)); pi=prev[r.dataset]
    p=r.r*pi/(r.r*pi+fpr*(1-pi)); f=2*p*r.r/(p+r.r); fs=2*r.p_s*r.r/(r.p_s+r.r)
    return pd.Series({'fpr':fpr,'p_nat':p,'f1_enr':fs,'f1_nat':f})
enr=pd.concat([enr,enr.apply(proj,axis=1)],axis=1); print(enr.round(3).to_string())
enr.to_csv('projection.csv',index=False)
