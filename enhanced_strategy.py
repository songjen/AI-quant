"""
Top30 策略增强版
- 技术派生特征：动量 + 加速度 + 交叉比率 + 截面排名
- 参数扫描：Top K = 5/10/20/30/50
- 回测指标：Sharpe / 最大回撤 / 胜率 / Calmar
- 可视化：策略对比雷达图 + 回撤曲线 + 特征重要性
"""
import warnings, os, json
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
RANDOM_STATE = 42
OUT_DIR = r'C:\Users\DELL\WorkBuddy\2026-07-17-13-03-09'

# ── 加载 ──
df = pd.read_csv(r'C:/Users/DELL/Desktop/model_data.csv', encoding='utf-8-sig')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Code', 'Date']).reset_index(drop=True)

BASE_COLS = [c for c in df.columns if c not in ['Date', 'Code', 'Next_Ret']]
print(f'基础因子: {len(BASE_COLS)}')

# ── 技术派生特征 ──
# 1) 动量：因子季度环比变化
mom_cols = []
for c in BASE_COLS:
    nc = f'{c}_mom'
    df[nc] = df.groupby('Code')[c].diff(1)
    mom_cols.append(nc)

# 2) 加速度：二阶差分
acc_cols = []
for c in BASE_COLS:
    nc = f'{c}_acc'
    df[nc] = df.groupby('Code')[c].diff(2)
    acc_cols.append(nc)

# 3) 截面排名（每期内所有股票的因子排名）
rank_cols = []
for c in BASE_COLS:
    nc = f'{c}_rank'
    df[nc] = df.groupby('Date')[c].rank(pct=True)
    rank_cols.append(nc)

# 4) 交叉比率
ratio_cols = []
pairs = [('MV', '企业倍数(EV除EBITDA)'), ('市盈率PE(TTM)', '市净率PB(MRQ)'), ('市销率PS(TTM)', '市盈率PE(TTM)')]
for a, b in pairs:
    if a in BASE_COLS and b in BASE_COLS:
        nc = f'{a}_div_{b}'
        df[nc] = df[a] / df[b].replace(0, np.nan)
        ratio_cols.append(nc)

ALL_COLS = BASE_COLS + mom_cols + acc_cols + rank_cols + ratio_cols
# 处理无穷和缺失
for c in ALL_COLS:
    df[c] = df[c].replace([np.inf, -np.inf], np.nan)

print(f'技术派生后总特征: {len(ALL_COLS)}')

# ── 参数扫描: Top K ──
TOP_K_VALUES = [5, 10, 20, 30, 50]
dates = sorted(df['Date'].unique())
train_dates = dates[:6]
test_dates = dates[6:]

models = {
    'LinearReg': LinearRegression(),
    'LogisticReg': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced'),
    'DecisionTree': DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),
}
model_types = {'LinearReg':'reg', 'LogisticReg':'cls', 'DecisionTree':'reg', 'RandomForest':'reg'}

# 清理缺失值行
df_clean = df.dropna(subset=ALL_COLS + ['Next_Ret']).copy()
print(f'清洗后样本: {len(df_clean)}')

# Walk-Forward 扫描
scan_results = []
for topk in TOP_K_VALUES:
    for mname, model in models.items():
        is_cls = model_types[mname] == 'cls'
        period_rets = []
        sp_list = []
        for i, test_date in enumerate(test_dates):
            train = df_clean[df_clean['Date'].isin(train_dates + test_dates[:i])]
            test = df_clean[df_clean['Date'] == test_date]
            X_tr = train[ALL_COLS].values
            y_tr = train['Next_Ret'].values
            X_te = test[ALL_COLS].values
            y_te = test['Next_Ret'].values
            y_tr_cls = (y_tr > 0).astype(int)

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            yy = y_tr_cls if is_cls else y_tr
            model.fit(X_tr_s, yy)
            pred_s = model.predict_proba(X_te_s)[:, 1] if is_cls else model.predict(X_te_s)

            # TopK 等权
            top_idx = np.argsort(-pred_s)[:topk]
            ew_ret = y_te[top_idx].mean()
            period_rets.append(ew_ret)
            sp, _ = spearmanr(pred_s, y_te)
            sp_list.append(sp)

        returns = np.array(period_rets)
        cum_ret = (1 + returns).prod() - 1
        sharpe = returns.mean() / returns.std() * np.sqrt(4) if returns.std() > 0 else 0
        max_dd = (returns.min() if returns.min() < 0 else 0)
        win_rate = (returns > 0).mean()
        scan_results.append({
            'model': mname, 'topk': topk,
            'total_return': cum_ret, 'sharpe': sharpe,
            'max_drawdown': max_dd, 'win_rate': win_rate,
            'avg_spearman': np.mean(sp_list),
            'period_returns': returns.tolist()
        })

# 基准
bmk_rets = []
for test_date in test_dates:
    test = df_clean[df_clean['Date'] == test_date]
    bmk_rets.append(test['Next_Ret'].mean())
bmk_rets = np.array(bmk_rets)
bmk_cum = (1 + bmk_rets).prod() - 1
bmk_sharpe = bmk_rets.mean() / bmk_rets.std() * np.sqrt(4) if bmk_rets.std() > 0 else 0

print(f'\n基准: 累计={bmk_cum:.2%} Sharpe={bmk_sharpe:.2f}')

sr_df = pd.DataFrame(scan_results)
# 找最佳
best_rf = sr_df[(sr_df['model']=='RandomForest')].sort_values('sharpe', ascending=False).iloc[0]
best_all = sr_df.sort_values('sharpe', ascending=False).iloc[0]
print(f'最佳: {best_all["model"]} TopK={best_all["topk"]} 收益={best_all["total_return"]:.2%} Sharpe={best_all["sharpe"]:.2f}')

# ── 可视化 ──
fig = plt.figure(figsize=(18, 12))
fig.suptitle('Top30 策略增强分析 —— 技术指标 + 参数扫描', fontsize=16, fontweight='bold')

gs = fig.add_gridspec(3, 4)

# 1) Sharpe vs TopK (各模型)
ax1 = fig.add_subplot(gs[0, :2])
for mname in ['LinearReg', 'LogisticReg', 'DecisionTree', 'RandomForest']:
    sub = sr_df[sr_df['model']==mname]
    ax1.plot(sub['topk'], sub['sharpe'], 'o-', lw=2, label=mname, markersize=6)
ax1.axhline(y=bmk_sharpe, color='gray', ls=':', lw=1.5, label=f'基准 Sharpe={bmk_sharpe:.2f}')
ax1.set_xlabel('Top K')
ax1.set_ylabel('Sharpe Ratio (年化)')
ax1.set_title('参数扫描：不同 Top K 下的 Sharpe 比率')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2) 累计收益 vs TopK
ax2 = fig.add_subplot(gs[0, 2:])
for mname in ['LinearReg', 'LogisticReg', 'DecisionTree', 'RandomForest']:
    sub = sr_df[sr_df['model']==mname]
    ax2.plot(sub['topk'], sub['total_return'], 'o-', lw=2, label=mname, markersize=6)
ax2.axhline(y=bmk_cum, color='gray', ls=':', lw=1.5, label=f'基准 {bmk_cum:.2%}')
ax2.set_xlabel('Top K')
ax2.set_ylabel('累计收益')
ax2.set_title('不同 Top K 下的累计收益')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)
ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.0%}'))

# 3) 雷达图（TopK=30 各模型对比）
ax3 = fig.add_subplot(gs[1, 0], projection='polar')
metrics = ['total_return', 'sharpe', 'win_rate', 'avg_spearman']
labels = ['累计收益', 'Sharpe', '胜率', '秩相关']
base30 = sr_df[sr_df['topk']==30]
angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist() + [0]
colors_m = {'LinearReg':'#378ADD','LogisticReg':'#534AB7','DecisionTree':'#1D9E75','RandomForest':'#D85A30'}
for mname in ['LinearReg', 'LogisticReg', 'DecisionTree', 'RandomForest']:
    row = base30[base30['model']==mname].iloc[0]
    vals = [max(0, row[m]) for m in metrics] + [max(0, row[metrics[0]])]
    # 归一化
    if max(vals) > 0:
        vals = [v/max(vals) for v in vals]
    ax3.plot(angles, vals, 'o-', color=colors_m[mname], lw=1.5, label=mname, markersize=4, alpha=0.8)
ax3.set_xticks(angles[:-1])
ax3.set_xticklabels(labels, fontsize=9)
ax3.set_title('TopK=30 各模型评估雷达图', fontsize=11)
ax3.legend(fontsize=7, loc='upper right', bbox_to_anchor=(1.3, 1.1))

# 4) 回撤曲线 (RF Top30)
ax4 = fig.add_subplot(gs[1, 1:3])
rf30 = sr_df[(sr_df['model']=='RandomForest') & (sr_df['topk']==30)].iloc[0]
rets = rf30['period_returns']
cumul = np.cumprod(1 + np.array(rets))
running_max = np.maximum.accumulate(cumul)
drawdown = (cumul - running_max) / running_max
periods = [str(d.date()) for d in test_dates]

ax4.fill_between(range(len(periods)), drawdown*100, 0, color='#d63031', alpha=0.3, label='回撤')
ax4.plot(range(len(periods)), drawdown*100, 'o-', color='#d63031', lw=1.5)
ax4.set_xticks(range(len(periods)))
ax4.set_xticklabels(periods, fontsize=9)
ax4.set_ylabel('回撤 (%)')
ax4.set_title(f'RF Top30 回撤曲线 (最大回撤: {rf30["max_drawdown"]:.2%})')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

# 5) 各模型收益对比热图
ax5 = fig.add_subplot(gs[1, 3])
pivot = sr_df[sr_df['topk']==30].pivot_table(
    index='model', values='total_return', aggfunc='first')
im = ax5.imshow(pivot.values.reshape(-1,1), cmap='RdYlGn', aspect='auto')
ax5.set_yticks(range(len(pivot.index)))
ax5.set_yticklabels(pivot.index, fontsize=9)
ax5.set_xticks([])
ax5.set_title('Top30 总收益', fontsize=10)
for i, v in enumerate(pivot.values):
    ax5.text(0, i, f'{v[0]:+.1%}', ha='center', va='center', fontsize=10, fontweight='bold',
             color='white' if abs(v[0]) > 0.15 else 'black')

# 6) 特征重要性 (RF)
ax6 = fig.add_subplot(gs[2, :])
# 训练一个全量 RF 来获取特征重要性
rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1)
all_train = df_clean[df_clean['Date'].isin(train_dates)]
X_all = all_train[ALL_COLS].values
y_all = all_train['Next_Ret'].values
scaler = StandardScaler()
X_all_s = scaler.fit_transform(X_all)
rf_model.fit(X_all_s, y_all)
imp = pd.Series(rf_model.feature_importances_, index=ALL_COLS).sort_values(ascending=False)

# 显示 Top20
top_imp = imp.head(20)
colors_imp = ['#D85A30' if 'mom' in n or 'acc' in n or 'rank' in n or 'div' in n else '#378ADD' for n in top_imp.index]
ax6.barh(range(len(top_imp)), top_imp.values, color=colors_imp, alpha=0.8)
ax6.set_yticks(range(len(top_imp)))
ax6.set_yticklabels(top_imp.index, fontsize=8)
ax6.invert_yaxis()
ax6.set_xlabel('Importance')
ax6.set_title('特征重要性 TOP20 (橙色=技术派生, 蓝色=基础因子)')
# 加图例
from matplotlib.patches import Patch
ax6.legend([Patch(color='#D85A30'), Patch(color='#378ADD')],
           ['技术派生特征', '基础财务因子'], fontsize=9, loc='lower right')

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'enhanced_analysis.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'增强分析图: {fig_path}')

# ── 保存 JSON ──
out = {
    'scan_summary': {
        'n_features_total': len(ALL_COLS),
        'n_features_base': len(BASE_COLS),
        'n_features_tech': len(ALL_COLS) - len(BASE_COLS),
        'benchmark_total_return': bmk_cum,
        'benchmark_sharpe': bmk_sharpe,
    },
    'best_config': {
        'model': best_all['model'], 'topk': int(best_all['topk']),
        'total_return': best_all['total_return'],
        'sharpe': best_all['sharpe'],
    },
    'results': [
        {k: (v if not isinstance(v, (np.ndarray, list)) else None)
         for k, v in r.items() if k != 'period_returns'}
        for r in scan_results
    ]
}
with open(os.path.join(OUT_DIR, 'enhanced_result.json'), 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)

print('\n═══════════ 参数扫描结果 (TOP 5) ═══════════')
print(f'{"模型":15s} {"TopK":5s} {"收益":>8s} {"Sharpe":>8s} {"回撤":>8s} {"胜率":>6s} {"Spearman":>9s}')
for _, r in sr_df.sort_values('sharpe', ascending=False).head(10).iterrows():
    print(f'{r["model"]:15s} {int(r["topk"]):5d} {r["total_return"]:>7.1%} {r["sharpe"]:>7.2f} '
          f'{r["max_drawdown"]:>7.1%} {r["win_rate"]:>5.0%} {r["avg_spearman"]:>8.3f}')

# ── 创建 HTML 报告 ──
html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head><meta charset="UTF-8">
<title>Top30 策略增强分析报告</title>
<style>
* {{margin:0;padding:0;box-sizing:border-box}}
body {{font-family:'Microsoft YaHei',sans-serif;background:#f5f6fa;color:#2d3436;padding:24px}}
h1 {{font-size:24px;margin-bottom:8px}}
h2 {{font-size:16px;color:#636e72;margin:20px 0 12px}}
.card {{background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06)}}
.grid {{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-bottom:20px}}
.stat {{text-align:center}}
.stat .v {{font-size:28px;font-weight:700}}
.stat .l {{font-size:12px;color:#636e72}}
.pos {{color:#d63031}} .neg {{color:#00b894}}
img {{max-width:100%;border-radius:8px;margin:12px 0}}
table {{width:100%;border-collapse:collapse;font-size:13px}}
th {{background:#f8f9fa;padding:8px 12px;border-bottom:2px solid #dfe6e9}}
td {{padding:8px 12px;border-bottom:1px solid #f1f2f6;text-align:center}}
</style></head>
<body>
<h1>Top30 策略增强分析报告</h1>
<p>模型: LR / Logistic / DT / RF · 技术派生特征: {len(ALL_COLS)-len(BASE_COLS)} 个 · 参数扫描: K=5~50</p>
<div class="grid">
  <div class="card stat"><div class="v pos">{best_all["total_return"]:.1%}</div><div class="l">最优配置累计收益</div></div>
  <div class="card stat"><div class="v" style="color:#6c5ce7">{best_all["sharpe"]:.2f}</div><div class="l">最优 Sharpe (年化)</div></div>
  <div class="card stat"><div class="v">{best_all["model"]}</div><div class="l">最优模型 · TopK={int(best_all["topk"])}</div></div>
  <div class="card stat"><div class="v">{len(ALL_COLS)}</div><div class="l">总特征数 (基础{len(BASE_COLS)}+技术{len(ALL_COLS)-len(BASE_COLS)})</div></div>
</div>
<h2>参数扫描结果 (按 Sharpe 排序)</h2>
<div class="card">
<table><thead><tr><th>模型</th><th>TopK</th><th>总收益</th><th>Sharpe</th><th>最大回撤</th><th>胜率</th><th>Spearman</th></tr></thead>
<tbody>{"".join(f"<tr><td>{r['model']}</td><td>{int(r['topk'])}</td><td class='{('pos' if r['total_return']>0 else 'neg')}'>{r['total_return']:.1%}</td><td>{r['sharpe']:.2f}</td><td class='neg'>{r['max_drawdown']:.1%}</td><td>{r['win_rate']:.0%}</td><td>{r['avg_spearman']:.3f}</td></tr>" for _, r in sr_df.sort_values('sharpe', ascending=False).iterrows())}</tbody></table></div>
<h2>分析图表</h2>
<div class="card"><img src="enhanced_analysis.png" alt="增强分析图"></div>
<div class="card"><img src="top30_overview.png" alt="策略总览"></div>
</body></html>'''

html_path = os.path.join(OUT_DIR, 'report_top30.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'HTML 报告: {html_path}')
print('\n✅ 增强分析完成！')
