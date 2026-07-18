# -*- coding: utf-8 -*-
"""3只股票回测v3 — 全市场分位数特征 + 对接akshare财务摘要"""
import os, warnings
import numpy as np
import pandas as pd
import akshare as ak
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings('ignore')

BASE = r'C:\Users\DELL\Desktop'
MODEL_DATA = os.path.join(BASE, 'model_data.csv')
OUTPUT = os.path.join(BASE, '量化交易', 'ai-quant', 'data')
os.makedirs(OUTPUT, exist_ok=True)

base_features = [
    '企业倍数(EV除EBITDA)', '市净率PB(MRQ)', '市现率PCF(现金净流量TTM)',
    '市现率PCF(经营现金流TTM)', '市盈率PE(TTM)', '市盈率PE(TTM,扣除非经常性损益)',
    '市销率PS(TTM)', '股息率(近12个月)', 'MV',
    '净利润同比增长率', '净资产同比增长率', '利润总额(同比增长率)',
    '基本每股收益(同比增长率)', '总资产同比增长率', '现金净流量同比增长率',
    '经营活动产生的现金流量净额(同比增长率)', '营业利润(同比增长率)',
    '营业总收入(同比增长率)', '营业收入(同比增长率)'
]

print("第1步: 训练模型")
df_all = pd.read_csv(MODEL_DATA)
df_all['Date_parsed'] = pd.to_datetime(df_all['Date'].apply(lambda d: d.replace('/', '-')))
dates_c = sorted(df_all['Date_parsed'].unique())
train_dates = dates_c[:7]
train_df = df_all[df_all['Date_parsed'].isin(train_dates)].copy()

# 训练数据特征
tr = train_df[['Date','Date_parsed','Code']+base_features+['Next_Ret']].copy()
for f in base_features:
    tr[f'rank_{f}'] = tr.groupby('Date')[f].rank(pct=True, ascending=True)
    g=tr.groupby('Date')[f]; z=f'z_{f}'
    tr[z]=(tr[f]-g.transform('mean'))/g.transform('std').replace(0,1); tr[z]=tr[z].clip(-5,5)
vf=['市净率PB(MRQ)','市盈率PE(TTM)','市盈率PE(TTM,扣除非经常性损益)','市销率PS(TTM)','市现率PCF(经营现金流TTM)']
for f in vf: tr[f'value_rank_{f}']=tr.groupby('Date')[f].rank(pct=True, ascending=False)
tr['value_score']=tr[[f'value_rank_{f}'for f in vf]].mean(1)
gf=['净利润同比增长率','营业总收入(同比增长率)','营业利润(同比增长率)','基本每股收益(同比增长率)']
for f in gf: tr[f'growth_rank_{f}']=tr.groupby('Date')[f].rank(pct=True, ascending=True)
tr['growth_score']=tr[[f'growth_rank_{f}'for f in gf]].mean(1)
tr['quality_rank_cf']=tr.groupby('Date')['经营活动产生的现金流量净额(同比增长率)'].rank(pct=True)
tr['quality_rank_equity']=tr.groupby('Date')['净资产同比增长率'].rank(pct=True)
tr['quality_score']=tr[['quality_rank_cf','quality_rank_equity']].mean(1)
tr['log_mv']=np.log(tr['MV'].clip(lower=1))
tr['rank_mv']=tr.groupby('Date')['log_mv'].rank(pct=True)
tr['interact_value_growth']=tr['value_score']*tr['growth_score']
tr['interact_value_quality']=tr['value_score']*tr['quality_score']
tr['interact_growth_quality']=tr['growth_score']*tr['quality_score']
tr['interact_size_value']=tr['rank_mv']*tr['value_score']
excl=['Date','Date_parsed','Code','Next_Ret']+base_features
fc=[c for c in tr.columns if c not in excl]

scaler=StandardScaler(); X_tr_s=scaler.fit_transform(tr[fc].values)
y_tr=tr['Next_Ret'].values
lr=LinearRegression(); lr.fit(X_tr_s, y_tr)
logreg=LogisticRegression(C=1.0,max_iter=1000,random_state=42)
logreg.fit(X_tr_s,(y_tr>0).astype(int))

# 训练集每个特征的分位数参考
feature_ref={}
for f in base_features:
    v=train_df[f].dropna().values
    feature_ref[f]=np.sort(v) if len(v)>100 else np.array([0.0,0.001,1.0])

print(f"模型就绪: {len(tr)}样本, {len(fc)}特征")

print("\n第2步: 获取3只股票数据")
codes={'600079':'ST人福','605507':'国邦医药','000756':'新华制药'}
stock_abs={}
for code,name in codes.items():
    print(f"  {name}...",end=' ')
    try:
        a=ak.stock_financial_abstract_ths(symbol=code, indicator='按报告期')
        a['报告期']=pd.to_datetime(a['报告期']); a=a.sort_values('报告期')
        stock_abs[code]=a; print(f"摘要{len(a)}条")
    except Exception as e:
        stock_abs[code]=pd.DataFrame(); print(f"❌ {e}")

print("\n第3步: 构造特征")
qends=pd.date_range('2018-03-31','2022-06-30',freq='QE')
rows=[]
for code,name in codes.items():
    ci=int(code); a=stock_abs[code]
    exist=df_all[df_all['Code']==ci]
    for qe in qends:
        r={'Code':ci,'股票':name,'Date':qe.strftime('%Y/%m/%d')}
        em=exist[exist['Date_parsed']==qe]
        if len(em)>0:
            for f in base_features:
                if f in em.columns and pd.notna(em.iloc[0][f]): r[f]=em.iloc[0][f]
        if len(a)>0:
            am=a[a['报告期']==qe]
            if len(am)>0:
                ar=am.iloc[0]
                for src,tgt in [('净利润同比增长率','净利润同比增长率'),('营业总收入同比增长率','营业总收入(同比增长率)')]:
                    v=ar.get(src)
                    if pd.notna(v) and v and v!='False':
                        try: r[tgt]=float(str(v).replace('%',''))
                        except: pass
        for f in base_features:
            if f not in r or pd.isna(r.get(f)):
                r[f]=train_df[f].median()
        rows.append(r)

sdf=pd.DataFrame(rows)
print(f"  构建: {len(sdf)}条")

# 特征工程（基于训练集全市场分位数）
def pct_of_val(val, arr):
    if pd.isna(val): return 0.5
    return min(max(np.searchsorted(arr,val)/len(arr),0.01),0.99)

for f in base_features:
    arr=feature_ref[f]
    sdf[f'rank_{f}']=sdf[f].apply(lambda v: pct_of_val(v,arr))
    sdf[f'z_{f}']=((sdf[f]-train_df[f].mean())/max(train_df[f].std(),1e-6)).clip(-5,5)

for f in vf: sdf[f'value_rank_{f}']=1-sdf[f'rank_{f}']
sdf['value_score']=sdf[[f'value_rank_{f}'for f in vf]].mean(1)
for f in gf: sdf[f'growth_rank_{f}']=sdf[f'rank_{f}']
sdf['growth_score']=sdf[[f'growth_rank_{f}'for f in gf]].mean(1)
sdf['quality_rank_cf']=sdf['rank_经营活动产生的现金流量净额(同比增长率)']
sdf['quality_rank_equity']=sdf['rank_净资产同比增长率']
sdf['quality_score']=sdf[['quality_rank_cf','quality_rank_equity']].mean(1)
sdf['log_mv']=np.log(sdf['MV'].clip(lower=1))
sdf['rank_mv']=sdf.groupby('Date')['log_mv'].rank(pct=True)
sdf['interact_value_growth']=sdf['value_score']*sdf['growth_score']
sdf['interact_value_quality']=sdf['value_score']*sdf['quality_score']
sdf['interact_growth_quality']=sdf['growth_score']*sdf['quality_score']
sdf['interact_size_value']=sdf['rank_mv']*sdf['value_score']

X_s=scaler.transform(sdf[fc].fillna(0).values)
sdf['LR_pred']=lr.predict(X_s)
sdf['LG_prob']=logreg.predict_proba(X_s)[:,1]

# 实际收益
for code,name in codes.items():
    ci=int(code); mask=sdf['Code']==ci
    sub_exist=df_all[df_all['Code']==ci][['Date_parsed','Next_Ret']]
    for i in sdf[mask].index:
        q=sdf.loc[i,'Date']; qd=pd.to_datetime(q.replace('/','-'))
        em=sub_exist[sub_exist['Date_parsed']==qd]
        sdf.loc[i,'实际收益']=em.iloc[0]['Next_Ret'] if len(em)>0 and pd.notna(em.iloc[0]['Next_Ret']) else np.nan

print("\n第4步: 回测")
print("="*60)
print(f"{'股票':10s} {'策略':10s} {'累计收益':>10s} {'年化收益':>10s} {'夏普':>7s} {'胜率':>5s} {'季度':>4s}")
print("-"*60)
all_res=[]
for code,name in codes.items():
    ci=int(code)
    sub=sdf[sdf['Code']==ci].sort_values('Date').reset_index(drop=True)
    rets=sub['实际收益'].values
    lr_s=(sub['LR_pred']>sub['LR_pred'].median()).astype(int).values
    lg_s=(sub['LG_prob']>0.5).astype(int).values
    
    for sn,sig in [('线性回归',lr_s),('逻辑回归',lg_s),('买入持有',None)]:
        sr=rets if sig is None else sig*rets
        v=sr[~np.isnan(sr)&(sr!=0)]
        if len(v)<2: continue
        cum=np.prod(1+v)-1; n=len(v)
        ann=(1+cum)**(4/n)-1; vol=np.std(v,ddof=1)*np.sqrt(4)
        sh=(ann-0.02)/vol if vol>0 else 0; win=np.mean(v>0)
        all_res.append({'股票':name,'策略':sn,'累计收益':cum,'年化收益':ann,'夏普':sh,'胜率':win,'季度':n})
        print(f"{name:10s} {sn:10s} {cum:+9.2%} {ann:+9.2%} {sh:7.2f} {win:4.0%} {n:4d}")
    print(f"  LR预测范围: {sub['LR_pred'].min():.3f}~{sub['LR_pred'].max():.3f}")
    print(f"  实际收益: {', '.join([f'{v:+.2%}' if not np.isnan(v) else 'N/A' for v in rets[:10]])}...")

res=pd.DataFrame(all_res)
res.to_csv(os.path.join(OUTPUT,'strategy_results_3stocks_v3.csv'),index=False,encoding='utf-8-sig')
sdf.to_csv(os.path.join(OUTPUT,'stock_3stocks_v3.csv'),index=False,encoding='utf-8-sig')
print(f"\n✅ 保存到 {OUTPUT}/")
