import pandas as pd, numpy as np, glob
import os
B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")+"/"
N={'creditcard':'Credit Card','ecommerce':'E-commerce','ieee':'IEEE-CIS','bank_fraud':'BAF (Base)','fraud_transactions':'Card Tx (Sim.)','paysim':'PaySim'}
order=['creditcard','ecommerce','ieee','bank_fraud','fraud_transactions','paysim']
M={'logistic_regression':'Logistic Regression','random_forest':'Random Forest','extra_trees':'Extra Trees','hist_gradient_boosting':'HistGradientBoosting','xgboost':'XGBoost','lightgbm':'LightGBM'}
f=lambda x:f"{x:.3f}"
out=[]
# T2
s12=pd.read_csv(B+"step12/step12_model_results.csv")
lr=pd.read_csv(B+"step03/step03_results.csv"); lr=lr[lr.model=='logreg_liblinear'].rename(columns={'avg_precision':'pr_auc'}); lr['model']='logistic_regression'
d=pd.concat([lr[['dataset','model','precision','recall','f1','pr_auc','roc_auc']],s12[['dataset','model','precision','recall','f1','pr_auc','roc_auc']]])
rows=[]
for ds in order:
    g=d[d.dataset==ds]; bf=g.f1.max(); bp=g.pr_auc.max(); first=True
    for _,r in g.iterrows():
        F=lambda v,b: ("\\textbf{"+f(v)+"}") if abs(v-b)<1e-9 else f(v)
        rows.append(f"{('\\multirow{'+str(len(g))+'}{*}{'+N[ds]+'}') if first else ''} & {M[r.model]} & {f(r.precision)} & {f(r.recall)} & {F(r.f1,bf)} & {F(r.pr_auc,bp)} & {f(r.roc_auc)} \\\\"); first=False
    rows.append("\\midrule")
rows=rows[:-1]
out.append("% T2\n"+"\n".join(rows))
# T3
s14=pd.read_csv(B+"step14/step14_all_results.csv")
rows=[]
for ds in order:
    g=s14[s14.dataset==ds]; mdl=g.model.iloc[0]
    cell=lambda st,met: (f(g[g.strategy==st][met].values[0]) if st in g.strategy.values else '--')
    rows.append(f"{N[ds]} & {M[mdl]} & "+" & ".join(cell(st,'f1') for st in ['baseline','class_weight','smote','adasyn'])+" & "+" & ".join(cell(st,'pr_auc') for st in ['baseline','class_weight','smote','adasyn'])+" \\\\")
out.append("% T3\n"+"\n".join(rows))
# T4
v=pd.read_csv(B+"step13/step13_all_variant_results.csv")
rows=[]
for ds in order:
    g=v[v.dataset==ds]; G=lambda vn: g[g.variant==vn].iloc[0]
    a,b,c,e=G('raw_default_0.50'),G('raw_tuned_f1'),G('calibrated_isotonic_0.50'),G('calibrated_isotonic_tuned_f1')
    rows.append(f"{N[ds]} & {M[a.model_name]} & {f(a.f1)} & {f(b.f1)} ({b.threshold:.2f}) & {f(c.f1)} & {f(e.f1)} ({e.threshold:.2f}) & {a.brier:.4f} & {c.brier:.4f} \\\\")
out.append("% T4\n"+"\n".join(rows))
# T5
s15=pd.read_csv(B+"step15/step15_all_results.csv")
rows=[]
for ds in order:
    g=s15[s15.dataset==ds]; bs=s12[s12.dataset==ds].sort_values('f1').iloc[-1]
    G=lambda m: g[g.model==m].iloc[0]
    sv,wv,st=G('soft_voting'),G('weighted_soft_voting'),G('stacking_logistic_meta')
    rows.append(f"{N[ds]} & {f(bs.f1)} ({M[bs.model]}) & {f(sv.default_f1)} & {f(wv.default_f1)} & {f(st.default_f1)} & {f(sv.default_pr_auc)} & {f(wv.default_pr_auc)} & {f(st.default_pr_auc)} & {f(wv.f1)} / {f(st.f1)} \\\\")
out.append("% T5\n"+"\n".join(rows))
# T6
p=pd.read_csv('projection.csv'); prev={'bank_fraud':0.0110,'creditcard':0.00167,'ecommerce':0.0221,'fraud_transactions':0.00579,'ieee':0.0350,'paysim':0.00129}
rows=[]
for ds in ['ieee','bank_fraud','fraud_transactions','paysim']:
    g=p[p.dataset==ds]; first=True
    for _,r in g.iterrows():
        nm={'soft':'Soft voting','weighted':'Weighted voting','stacking':'Stacking'}[r.ens]
        rows.append(f"{('\\multirow{3}{*}{'+N[ds]+'}') if first else ''} & {nm} & {100*r.pi_s:.1f}\\% & {100*prev[ds]:.2f}\\% & {f(r.p_s)} & {f(r.r)} & {f(r.f1_enr)} & {f(r.p_nat)} & {f(r.f1_nat)} \\\\"); first=False
    rows.append("\\midrule")
out.append("% T6\n"+"\n".join(rows[:-1]))
# T7 shap
rows=[]
for ds in order:
    sh=pd.read_csv(B+f"step12/step12_shap_{ds}_xgboost.csv").head(5)
    feats=[x.replace('num__','').replace('cat__','').replace('_','\\_') for x in sh.feature]
    rows.append(f"{N[ds]} & "+", ".join(feats)+" \\\\")
out.append("% T7\n"+"\n".join(rows))
open('gen_tables.txt','w').write("\n\n".join(out)); print("\n\n".join(out))
