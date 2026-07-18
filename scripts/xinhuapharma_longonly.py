# -*- coding: utf-8 -*-
"""新华制药 — 4模型纯多头策略回测（A股无做空机制）"""
import os, warnings, json, numpy as np, pandas as pd
import akshare as ak
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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

print("第1步: 训练4个模型")
df_all = pd.read_csv(MODEL_DATA)
df_all['Date_parsed'] = pd.to_datetime(df_all['Date'].apply(lambda d: d.replace('/', '-')))
dates_c = sorted(df_all['Date_parsed'].unique())
train_dates = dates_c[:7]
train_df = df_all[df_all['Date_parsed'].isin(train_dates)].copy()

# 特征工程
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

X_tr=tr[fc].values; y_tr=tr['Next_Ret'].values
scaler=StandardScaler(); X_tr_s=scaler.fit_transform(X_tr)

# 4个模型
models = {
    'LinearRegression': LinearRegression(),
    'DecisionTree': DecisionTreeRegressor(max_depth=6, min_samples_leaf=50, min_samples_split=100, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=30, n_jobs=-1, random_state=42),
    'LogisticRegression': LogisticRegression(C=1.0, max_iter=1000, random_state=42),
}
for name, m in models.items():
    if name == 'LogisticRegression':
        m.fit(X_tr_s, (y_tr>0).astype(int))
    else:
        m.fit(X_tr_s, y_tr)
print(f"训练完成: {len(X_tr)}样本, {len(fc)}特征")

# 第2步: 加载新华制药数据
print("\n第2步: 加载新华制药数据")
abs_df = ak.stock_financial_abstract_ths(symbol='000756', indicator='按报告期')
abs_df['报告期'] = pd.to_datetime(abs_df['报告期'])
abs_df = abs_df[abs_df['报告期'] >= '2018'].sort_values('报告期')

# 第3步: 构造季度特征
qends=pd.date_range('2018-03-31','2022-06-30',freq='QE')
exist=df_all[df_all['Code']==756]
rows=[]
for qe in qends:
    r={'Code':756,'Date':qe.strftime('%Y/%m/%d')}
    em=exist[exist['Date_parsed']==qe]
    if len(em)>0:
        for f in base_features:
            if f in em.columns and pd.notna(em.iloc[0][f]): r[f]=em.iloc[0][f]
    am=abs_df[abs_df['报告期']==qe]
    if len(am)>0:
        for src,tgt in [('净利润同比增长率','净利润同比增长率'),('营业总收入同比增长率','营业总收入(同比增长率)')]:
            v=am.iloc[0].get(src)
            if pd.notna(v) and v!='' and v!='False':
                try: r[tgt]=float(str(v).replace('%',''))
                except: pass
    for f in base_features:
        if f not in r or pd.isna(r.get(f)):
            r[f]=train_df[f].median()
    rows.append(r)
xh=pd.DataFrame(rows)

# 特征工程（基于训练集分位数）
def pct_of_val(v,arr):
    if pd.isna(v): return 0.5
    return min(max(np.searchsorted(arr,v)/len(arr),0.01),0.99)

feature_ref={}
for f in base_features:
    arr=np.sort(train_df[f].dropna().values) if len(train_df[f].dropna())>100 else np.array([0.0,1.0])
    feature_ref[f]=arr

for f in base_features:
    xh[f'rank_{f}']=xh[f].apply(lambda v:pct_of_val(v,feature_ref[f]))
    xh[f'z_{f}']=((xh[f]-train_df[f].mean())/max(train_df[f].std(),1e-6)).clip(-5,5)
for f in vf: xh[f'value_rank_{f}']=1-xh[f'rank_{f}']
xh['value_score']=xh[[f'value_rank_{f}'for f in vf]].mean(1)
for f in gf: xh[f'growth_rank_{f}']=xh[f'rank_{f}']
xh['growth_score']=xh[[f'growth_rank_{f}'for f in gf]].mean(1)
xh['quality_rank_cf']=xh['rank_经营活动产生的现金流量净额(同比增长率)']
xh['quality_rank_equity']=xh['rank_净资产同比增长率']
xh['quality_score']=xh[['quality_rank_cf','quality_rank_equity']].mean(1)
xh['log_mv']=np.log(xh['MV'].clip(lower=1))
xh['rank_mv']=xh.groupby('Date')['log_mv'].rank(pct=True)
xh['interact_value_growth']=xh['value_score']*xh['growth_score']
xh['interact_value_quality']=xh['value_score']*xh['quality_score']
xh['interact_growth_quality']=xh['growth_score']*xh['quality_score']
xh['interact_size_value']=xh['rank_mv']*xh['value_score']

X_xh=scaler.transform(xh[fc].fillna(0).values)

# 预测
for name in ['LinearRegression','DecisionTree','RandomForest']:
    xh[f'pred_{name}']=models[name].predict(X_xh)
xh['pred_LogisticRegression']=models['LogisticRegression'].predict_proba(X_xh)[:,1]

# 实际收益（从model_data提取）
exist_ret=df_all[df_all['Code']==756][['Date_parsed','Next_Ret']]
rets=[]
for _,row in xh.iterrows():
    q=row['Date']; qd=pd.to_datetime(q.replace('/','-'))
    em=exist_ret[exist_ret['Date_parsed']==qd]
    rets.append(em.iloc[0]['Next_Ret'] if len(em)>0 and pd.notna(em.iloc[0]['Next_Ret']) else np.nan)
xh['实际收益']=rets

# 第4步: 纯多头回测（A股无做空）
print("\n第4步: 纯多头策略回测")
model_labels={'LinearRegression':'线性回归','DecisionTree':'决策树','RandomForest':'随机森林','LogisticRegression':'逻辑回归'}

def long_only_backtest(rets, signals, model_name):
    """纯多头策略：信号1=持仓/0=现金"""
    srets=[]
    for i in range(len(rets)):
        if i==0: continue  # 第1期无信号
        sig=signals[i-1]  # 用上期信号交易本期
        if pd.isna(rets[i]): continue
        ret=rets[i]*sig  # 信号1=持仓,0=现金(0收益)
        srets.append(ret)
    return np.array(srets)

results={}
for name in ['LinearRegression','DecisionTree','RandomForest','LogisticRegression']:
    pred_col=f'pred_{name}'
    preds=xh[pred_col].values

    if name=='LogisticRegression':
        signals=(preds>0.5).astype(int)  # 概率>0.5做多
    else:
        signals=(preds>np.nanmedian(preds[pd.notna(xh['实际收益'])])).astype(int)  # >中位数做多

    srets=long_only_backtest(xh['实际收益'].values, signals, name)
    valid=srets[~np.isnan(srets)]

    if len(valid)>=2:
        cum=np.prod(1+valid)-1
        n=len(valid)
        ann=(1+cum)**(4/n)-1
        vol=np.std(valid,ddof=1)*np.sqrt(4)
        sharpe=(ann-0.02)/vol if vol>0 else 0
        win=np.mean(valid>0)
        max_dd=(np.cumprod(1+valid)/np.maximum.accumulate(np.cumprod(1+valid))-1).min()
    else:
        cum=ann=sharpe=win=max_dd=0; n=0

    results[name]={
        'label':model_labels[name],'cum':cum,'ann':ann,'sharpe':sharpe,
        'win':win,'max_dd':max_dd,'n':n,'rets':valid,'signals':signals
    }

# 买入持有
bh_rets=xh['实际收益'].values[1:]; bh_valid=bh_rets[~np.isnan(bh_rets)]
if len(bh_valid)>=2:
    bh_cum=np.prod(1+bh_valid)-1
    bh_ann=(1+bh_cum)**(4/len(bh_valid))-1
    bh_vol=np.std(bh_valid,ddof=1)*np.sqrt(4)
    bh_sharpe=(bh_ann-0.02)/bh_vol if bh_vol>0 else 0
    bh_win=np.mean(bh_valid>0)
    bh_max_dd=(np.cumprod(1+bh_valid)/np.maximum.accumulate(np.cumprod(1+bh_valid))-1).min()
else:
    bh_cum=bh_ann=bh_sharpe=bh_win=bh_max_dd=0
results['BH']={'label':'买入持有','cum':bh_cum,'ann':bh_ann,'sharpe':bh_sharpe,'win':bh_win,'max_dd':bh_max_dd,'n':len(bh_valid),'rets':bh_valid}

# 输出
print("="*70)
print(f"{'模型':12s} {'累计收益':>10s} {'年化收益':>10s} {'夏普':>7s} {'胜率':>6s} {'最大回撤':>10s} {'信号次数':>8s}")
print("-"*70)
for name in ['DecisionTree','LinearRegression','RandomForest','LogisticRegression','BH']:
    r=results[name]; sig_cnt=int(np.sum(r.get('signals',np.array([0]))))
    sig_str=f"{sig_cnt}/{len(r.get('signals',[]))}" if name!='BH' else '—'
    print(f"{r['label']:12s} {r['cum']:+9.2%} {r['ann']:+9.2%} {r['sharpe']:7.2f} {r['win']:5.0%} {r['max_dd']:9.2%} {sig_str:>8s}")

# 打印各季度明细
print(f"\n{'='*70}")
print("各季度信号明细:")
print(f"{'季度':12s} {'实际收益':>10s} {'线性回归':>10s} {'决策树':>10s} {'随机森林':>10s} {'逻辑回归':>10s}")
print("-"*70)
for i,row in xh.iterrows():
    if i==0:
        print(f"{row['Date']:12s} {row['实际收益']:>+9.2%}   {'(起始)':>10s} {'(起始)':>10s} {'(起始)':>10s} {'(起始)':>10s}")
        continue
    ret=row['实际收益']
    lr_s='做多' if results['LinearRegression']['signals'][i-1] else '现金'
    dt_s='做多' if results['DecisionTree']['signals'][i-1] else '现金'
    rf_s='做多' if results['RandomForest']['signals'][i-1] else '现金'
    lg_s='做多' if results['LogisticRegression']['signals'][i-1] else '现金'
    print(f"{row['Date']:12s} {ret:+9.2%} {lr_s:>10s} {dt_s:>10s} {rf_s:>10s} {lg_s:>10s}")

# 保存
output_json={
    'meta':{'stock':'新华制药','code':'000756','updated':pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')},
    'strategy_results':{
        name:{
            'label':r['label'],'cum_return':round(r['cum'],4),'ann_return':round(r['ann'],4),
            'sharpe':round(r['sharpe'],4),'win_rate':round(r['win'],4),'max_dd':round(r['max_dd'],4)
        } for name,r in results.items()
    },
    'quarterly_detail':[
        {
            'period':xh.iloc[i]['Date'],
            'actual_return':round(float(xh.iloc[i]['实际收益']),4) if pd.notna(xh.iloc[i]['实际收益']) else None,
            'LR_signal':int(results['LinearRegression']['signals'][i-1]) if i>0 else 0,
            'DT_signal':int(results['DecisionTree']['signals'][i-1]) if i>0 else 0,
            'RF_signal':int(results['RandomForest']['signals'][i-1]) if i>0 else 0,
            'LG_signal':int(results['LogisticRegression']['signals'][i-1]) if i>0 else 0,
        } for i in range(len(xh))
    ]
}
with open(os.path.join(OUTPUT,'新华制药_longonly.json'),'w',encoding='utf-8') as f:
    json.dump(output_json,f,ensure_ascii=False,indent=2)
xh.to_csv(os.path.join(OUTPUT,'新华制药_longonly_data.csv'),index=False,encoding='utf-8-sig')
print(f"\n✅ 结果保存到 {OUTPUT}/")
