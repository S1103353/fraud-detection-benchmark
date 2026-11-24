import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'serif','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
import os
B=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..","results")+"/"
names={'creditcard':'Credit Card','ecommerce':'E-commerce','ieee':'IEEE-CIS','bank_fraud':'Bank Account (BAF)','fraud_transactions':'Card Transactions (Sim.)','paysim':'PaySim'}
order=['paysim','creditcard','ecommerce','fraud_transactions','ieee','bank_fraud']
prev={'bank_fraud':0.0110,'creditcard':0.00167,'ecommerce':0.0221,'fraud_transactions':0.00579,'ieee':0.0350,'paysim':0.00129}
s12=pd.read_csv(B+"step12/step12_model_results.csv")
lr=pd.read_csv(B+"step03/step03_results.csv"); lr=lr[lr.model=='logreg_liblinear'].rename(columns={'avg_precision':'pr_auc'}); lr['model']='logistic_regression'
d=pd.concat([s12[['dataset','model','pr_auc','f1']],lr[['dataset','model','pr_auc','f1']]])
mk={'logistic_regression':('x','Logistic Regression'),'random_forest':('s','Random Forest'),'extra_trees':('^','Extra Trees'),'hist_gradient_boosting':('v','HistGB'),'xgboost':('o','XGBoost'),'lightgbm':('D','LightGBM')}
fig,ax=plt.subplots(1,2,figsize=(7.2,3.0),sharey=True)
for j,met in enumerate(['pr_auc','f1']):
    for i,ds in enumerate(order):
        g=d[d.dataset==ds]
        for _,r in g.iterrows():
            m,l=mk[r.model]; ax[j].scatter(r[met],i,marker=m,s=26,facecolor='none' if m not in 'x' else None,edgecolor='k' if m!='x' else None,color='k' if m=='x' else None,linewidth=0.8,label=l)
        if met=='pr_auc': ax[j].plot([prev[ds]]*2,[i-0.3,i+0.3],color='#C0392B',lw=1.2)
    ax[j].set_xlim(0,1); ax[j].set_yticks(range(len(order))); ax[j].set_yticklabels([names[o] for o in order]); ax[j].grid(axis='x',ls=':',lw=0.5)
    ax[j].set_xlabel('PR-AUC (red bar = fraud prevalence)' if met=='pr_auc' else 'F1 at threshold 0.5')
h,l=ax[0].get_legend_handles_labels(); u=dict(zip(l,h)); fig.legend(u.values(),u.keys(),loc='upper center',ncol=6,frameon=False,fontsize=7.5,bbox_to_anchor=(0.55,1.04))
ax[0].invert_yaxis(); fig.tight_layout(rect=(0,0,1,0.93)); fig.savefig('fig1_dataset_vs_model.pdf',bbox_inches='tight'); fig.savefig('fig1.png',dpi=130,bbox_inches='tight')
# fig2 calibration / threshold
v=pd.read_csv(B+"step13/step13_all_variant_results.csv")
var=[('raw_default_0.50','Cost-sensitive, raw, t=0.5'),('raw_tuned_f1','Raw, t tuned on calibration split'),('calibrated_isotonic_0.50','Isotonic-calibrated, t=0.5')]
fig,ax=plt.subplots(figsize=(7.2,2.6)); w=0.26; cols=['#bbbbbb','#555555','#C0392B']
for k,(vn,lab) in enumerate(var):
    vals=[v[(v.dataset==ds)&(v.variant==vn)].f1.values[0] for ds in order]
    ax.bar(np.arange(len(order))+(k-1)*w,vals,w,label=lab,color=cols[k])
ax.set_xticks(range(len(order))); ax.set_xticklabels([names[o] for o in order],fontsize=8); ax.set_ylabel('Test F1'); ax.set_ylim(0,1); ax.legend(frameon=False,fontsize=7.5,ncol=3,loc='upper center',bbox_to_anchor=(0.5,1.15)); ax.grid(axis='y',ls=':',lw=0.5)
fig.tight_layout(); fig.savefig('fig2_calibration.pdf',bbox_inches='tight'); fig.savefig('fig2.png',dpi=130,bbox_inches='tight')
# fig3 prevalence trap
p=pd.read_csv('projection.csv'); p=p[p.ens=='weighted']
s15=pd.read_csv(B+"step15/step15_all_results.csv"); s15=s15[s15.model=='weighted_soft_voting']
o3=['paysim','fraud_transactions','ieee','bank_fraud']
fig,ax=plt.subplots(figsize=(7.2,2.6)); w=0.26
a=[p[p.dataset==ds].f1_enr.values[0] for ds in o3]; b=[p[p.dataset==ds].f1_nat.values[0] for ds in o3]; c=[s15[s15.dataset==ds].default_f1.values[0] for ds in o3]
ax.bar(np.arange(4)-w,a,w,color='#bbbbbb',label='Fraud-enriched train, enriched test (reported)')
ax.bar(np.arange(4),b,w,color='#C0392B',label='Fraud-enriched train, projected to natural prevalence')
ax.bar(np.arange(4)+w,c,w,color='#555555',label='Natural-prevalence train/test (full data)')
ax.set_xticks(range(4)); ax.set_xticklabels([names[o] for o in o3],fontsize=8); ax.set_ylabel('F1 (weighted voting, t=0.5)'); ax.set_ylim(0,1.05); ax.legend(frameon=False,fontsize=7.2,ncol=1,loc='upper right'); ax.grid(axis='y',ls=':',lw=0.5)
fig.tight_layout(); fig.savefig('fig3_prevalence.pdf',bbox_inches='tight'); fig.savefig('fig3.png',dpi=130,bbox_inches='tight')
print([round(x,3) for x in a],[round(x,3) for x in b],[round(x,3) for x in c])
