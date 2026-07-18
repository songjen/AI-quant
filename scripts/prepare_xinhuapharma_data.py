# -*- coding: utf-8 -*-
"""新华制药可视化数据准备"""
import os, json, warnings
import numpy as np
import pandas as pd
import akshare as ak
warnings.filterwarnings('ignore')


def _parse(v):
    if pd.isna(v) or v == '' or v == 'False':
        return None
    try:
        return round(float(str(v).replace('%', '')), 2)
    except (ValueError, TypeError):
        return None

BASE = r'C:\Users\DELL\Desktop\量化交易\ai-quant'
DATA_OUT = os.path.join(BASE, 'data')
os.makedirs(DATA_OUT, exist_ok=True)

# 1. 加载v3预测数据
v3 = pd.read_csv(os.path.join(DATA_OUT, 'stock_3stocks_v3.csv'))
xh = v3[v3['股票']=='新华制药'].sort_values('Date').reset_index(drop=True)

# 2. 从model_data获取实际收益数据
model_df = pd.read_csv(r'C:\Users\DELL\Desktop\model_data.csv')
mxh = model_df[model_df['Code']==756][['Date','Next_Ret']].copy()
mxh['Date'] = mxh['Date'].str.replace('/','-')

# 3. 获取财务摘要（增长率）
abs_df = ak.stock_financial_abstract_ths(symbol='000756', indicator='按报告期')
abs_df['报告期'] = pd.to_datetime(abs_df['报告期'])
abs_df = abs_df[abs_df['报告期'] >= '2018'].sort_values('报告期')

# 4. 获取新浪财务数据（资产负债表关键项）
try:
    sina = ak.stock_financial_report_sina(stock='000756')
    sina['报告日'] = pd.to_datetime(sina['报告日'])
    sina = sina[sina['报告日'] >= '2018'].sort_values('报告日')
except:
    sina = pd.DataFrame()

# 5. 构建输出JSON
quarters = xh['Date'].tolist()
grow_rates = ['净利润同比增长率','营业总收入(同比增长率)']

output = {
    "meta": {
        "name": "新华制药",
        "code": "000756",
        "market": "SZ",
        "industry": "化学制药",
        "updated": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'),
        "description": "新华制药（000756.SZ）是一家以化学原料药和制剂为主的制药企业"
    },
    "predictions": {},
    "financial": {},
    "signals": {},
    "feature_history": {}
}

# 预测数据
preds = []
for _, row in xh.iterrows():
    d = row['Date']
    preds.append({
        "period": d,
        "LR_pred": round(float(row['LR_pred']), 4),
        "LG_prob": round(float(row['LG_prob']), 4),
        "actual_return": round(float(row['实际收益']), 4) if pd.notna(row['实际收益']) else None,
        "LG_signal": 1 if row['LG_prob'] > 0.5 else 0,
        "LR_signal": 1 if row['LR_pred'] > xh['LR_pred'].median() else 0
    })

# 补充实际收益（从model_data）
mxh_dict = dict(zip(mxh['Date'], mxh['Next_Ret']))
for p in preds:
    d = p['period'].replace('/','-')
    if p['actual_return'] is None and d in mxh_dict:
        p['actual_return'] = round(float(mxh_dict[d]), 4)

output['predictions'] = preds

# 财务数据
fin = []
for _, row in abs_df.iterrows():
    d = row['报告期'].strftime('%Y/%m/%d')
    fin.append({
        "period": d,
        "net_profit_yoy": _parse(row.get('净利润同比增长率')),
        "revenue_yoy": _parse(row.get('营业总收入同比增长率')),
        "eps": _parse(row.get('基本每股收益')),
        "bps": _parse(row.get('每股净资产')),
        "roe": _parse(row.get('净资产收益率')),
        "net_profit_margin": _parse(row.get('销售净利率')),
        "debt_ratio": _parse(row.get('资产负债率')),
    })
output['financial'] = fin

# 特征历史（关键特征的时间序列）
key_features = ['市净率PB(MRQ)','市盈率PE(TTM)','市销率PS(TTM)','MV',
                '净利润同比增长率','营业总收入(同比增长率)','value_score','growth_score']
feat_hist = []
for _, row in xh.iterrows():
    d = row['Date']
    fe = {"period": d}
    for f in key_features:
        v = row.get(f)
        fe[f] = round(float(v), 4) if pd.notna(v) else None
    feat_hist.append(fe)
output['feature_history'] = feat_hist

# 策略收益对比
output['strategy_comparison'] = {
    "线性回归": {"cum_return": -0.1756, "ann_return": -0.1045, "sharpe": -0.31, "win_rate": 0.29},
    "逻辑回归": {"cum_return": 1.3237, "ann_return": 0.4546, "sharpe": 0.35, "win_rate": 0.44},
    "买入持有": {"cum_return": 0.5327, "ann_return": 0.1863, "sharpe": 0.14, "win_rate": 0.40}
}

# 写入
with open(os.path.join(DATA_OUT, '新华制药_data.json'), 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, indent=2)
print(f"✅ 数据已保存 ({len(preds)}期预测, {len(fin)}期财务)")

