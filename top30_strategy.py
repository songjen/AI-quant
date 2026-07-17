"""
机器学习选股模型 — 季度 Top 30 策略
4 模型：线性回归(Baseline) / 逻辑回归 / 决策树 / 随机森林
等权 vs 预测值加权 对比
Time split: 训练 2020Q1-2021Q2 (6季) → 测试 2021Q3-2022Q2 (4季)
"""
import warnings, os, json
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

RANDOM_STATE = 42
DATA_PATH = r'C:/Users/DELL/Desktop/model_data.csv'
OUT_DIR = r'C:\Users\DELL\WorkBuddy\2026-07-17-13-03-09'
TOP_K = 30

# ── 加载数据 ──
df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(['Code', 'Date']).reset_index(drop=True)

# 特征列（去掉 Date, Code, Next_Ret）
FEATURE_COLS = [c for c in df.columns if c not in ['Date', 'Code', 'Next_Ret']]
print(f'特征数: {len(FEATURE_COLS)}')
print(f'样本数: {len(df)}')
print(f'日期: {sorted(df["Date"].dt.date.unique())}')

# ── 时间切分 ──
dates = sorted(df['Date'].unique())
train_dates = dates[:6]   # 2020Q1-2021Q2
test_dates = dates[6:]    # 2021Q3-2022Q2
print(f'\n训练期: {[d.date() for d in train_dates]}')
print(f'测试期: {[d.date() for d in test_dates]}')

# ── 定义模型 ──
models = {
    'LinearReg': LinearRegression(),
    'LogisticReg': LogisticRegression(max_iter=2000, random_state=RANDOM_STATE, class_weight='balanced'),
    'DecisionTree': DecisionTreeRegressor(max_depth=8, min_samples_leaf=10, random_state=RANDOM_STATE),
    'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=10, min_samples_leaf=10, random_state=RANDOM_STATE, n_jobs=-1),
}
model_types = {'LinearReg':'reg', 'LogisticReg':'cls', 'DecisionTree':'reg', 'RandomForest':'reg'}

# ── Walk-Forward 回测 ──
all_periods = []
for i, test_date in enumerate(test_dates):
    current_train = df[df['Date'].isin(train_dates + test_dates[:i])]
    test = df[df['Date'] == test_date]

    X_train = current_train[FEATURE_COLS].values
    y_train = current_train['Next_Ret'].values
    X_test = test[FEATURE_COLS].values
    y_test = test['Next_Ret'].values

    # 对逻辑回归：Y 二值化
    y_train_cls = (y_train > 0).astype(int)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    period_result = {'test_date': test_date.date(), 'n_train': len(X_train), 'n_test': len(X_test)}

    for mname, model in models.items():
        is_cls = model_types[mname] == 'cls'
        yy = y_train_cls if is_cls else y_train
        model.fit(X_train_s, yy)

        if is_cls:
            pred_prob = model.predict_proba(X_test_s)[:, 1]
            pred_score = pred_prob
        else:
            pred_score = model.predict(X_test_s)

        # 排序选择 Top K
        order = np.argsort(-pred_score)
        top_idx = order[:TOP_K]
        top_pred = pred_score[top_idx]
        top_actual = y_test[top_idx]

        # 等权收益
        ew_ret = top_actual.mean()
        # 预测加权收益（权重 ∝ max(0, pred_score)）
        w = np.maximum(0, top_pred)
        if w.sum() > 0:
            w = w / w.sum()
            pw_ret = (top_actual * w).sum()
        else:
            pw_ret = ew_ret

        # 选股能力
        sp, _ = spearmanr(pred_score, y_test)
        mse = mean_squared_error(y_test, pred_score) if not is_cls else np.nan

        period_result[f'{mname}_score'] = pred_score.tolist()
        period_result[f'{mname}_top_codes'] = test.iloc[top_idx]['Code'].tolist()
        period_result[f'{mname}_ew'] = ew_ret
        period_result[f'{mname}_pw'] = pw_ret
        period_result[f'{mname}_spearman'] = sp
        period_result[f'{mname}_mse'] = mse

    period_result['benchmark'] = y_test.mean()
    all_periods.append(period_result)
    print(f'Test {test_date.date()}: bmk={y_test.mean():.4f}', end=' ')
    for mname in models:
        print(f'{mname}_ew={period_result[f"{mname}_ew"]:.4f}', end=' ')
    print()

# ── 汇总结果 ──
results_df = pd.DataFrame(all_periods)
print('\n═══════ 策略对比 ═══════')
print(f'{"模型":15s} {"加权":6s} {"总收益":>8s} {"超额":>8s} {"Spearman":>10s}')
for mname in models:
    for wt, label in [('ew', '等权'), ('pw', '加权')]:
        col = f'{mname}_{wt}'
        total_ret = results_df[col].sum()
        total_excess = results_df[col].sum() - results_df['benchmark'].sum()
        avg_sp = results_df[f'{mname}_spearman'].mean()
        print(f'{mname:15s} {label:6s} {total_ret:>8.2%} {total_excess:>8.2%} {avg_sp:>10.4f}')

# ── 可视化 ──
FIGSIZE = (16, 12)
fig, axes = plt.subplots(2, 3, figsize=FIGSIZE)
fig.suptitle('机器学习选股模型 — 季度 Top 30 策略回测', fontsize=16, fontweight='bold')

dates_str = [str(d['test_date']) for d in all_periods]
x = np.arange(len(dates_str))
colors_m = {'LinearReg':'#378ADD', 'LogisticReg':'#534AB7', 'DecisionTree':'#1D9E75', 'RandomForest':'#D85A30'}
linestyle = {'ew': '-', 'pw': '--'}

# 1a. 累计净值
ax = axes[0, 0]
for mname in models:
    for wt, ls in [('ew', '-'), ('pw', '--')]:
        col = f'{mname}_{wt}'
        cum = (1 + results_df[col]).cumprod()
        label = f'{mname}({"等权" if wt=="ew" else "加权"})'
        ax.plot(dates_str, cum, ls=ls, color=colors_m[mname], lw=1.5, label=label, alpha=0.8)
bmk_cum = (1 + results_df['benchmark']).cumprod()
ax.plot(dates_str, bmk_cum, 'k:', lw=2, label='基准(全市场等权)')
ax.set_ylabel('净值')
ax.set_title('累计净值曲线')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

# 1b. 各期超额收益（等权）
ax = axes[0, 1]
for mname in models:
    excess = results_df[f'{mname}_ew'] - results_df['benchmark']
    ax.plot(dates_str, excess, 'o-', color=colors_m[mname], lw=1.5, label=mname, markersize=5)
ax.axhline(y=0, color='gray', ls=':', lw=0.5)
ax.set_ylabel('超额收益')
ax.set_title('各模型等权策略超额收益')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1c. 等权 vs 加权对比 (RF 为例)
ax = axes[0, 2]
ax.plot(dates_str, results_df['RandomForest_ew'], 'o-', color='#D85A30', lw=2, label='RF 等权', markersize=6)
ax.plot(dates_str, results_df['RandomForest_pw'], 's--', color='#e17055', lw=2, label='RF 预测加权', markersize=6)
ax.plot(dates_str, results_df['benchmark'], 'k:', lw=1.5, label='基准')
ax.set_ylabel('收益率')
ax.set_title('随机森林：等权 vs 预测加权 vs 基准')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1d. Spearman 秩相关
ax = axes[1, 0]
for mname in models:
    ax.plot(dates_str, results_df[f'{mname}_spearman'], 'o-', color=colors_m[mname], lw=1.5, label=mname, markersize=5)
ax.axhline(y=0, color='gray', ls=':', lw=0.5)
ax.set_ylabel('Spearman ρ')
ax.set_title('预测收益排序 vs 实际收益排序 秩相关')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 1e. 总收益对比柱状图
ax = axes[1, 1]
models_list = list(models.keys())
ew_totals = [results_df[f'{m}_ew'].sum() for m in models_list]
pw_totals = [results_df[f'{m}_pw'].sum() for m in models_list]
bmk_total = results_df['benchmark'].sum()
x_m = np.arange(len(models_list))
w = 0.3
bars1 = ax.bar(x_m - w/2, ew_totals, w, label='等权', color='#378ADD', alpha=0.8)
bars2 = ax.bar(x_m + w/2, pw_totals, w, label='预测加权', color='#D85A30', alpha=0.8)
ax.axhline(y=bmk_total, color='black', ls=':', lw=1.5, label=f'基准({bmk_total:.2%})')
ax.set_xticks(x_m)
ax.set_xticklabels(models_list, fontsize=9)
ax.set_ylabel('累计收益')
ax.set_title('各模型累计收益对比（4 个测试期）')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis='y')

# 1f. 综合指标表
ax = axes[1, 2]
ax.axis('off')
text = ['回测综合指标', '─'*22]
for mname in models:
    ew = results_df[f'{mname}_ew'].sum()
    pw = results_df[f'{mname}_pw'].sum()
    exc = results_df[f'{mname}_ew'].sum() - results_df['benchmark'].sum()
    sp = results_df[f'{mname}_spearman'].mean()
    text.append(f'{mname:15s}')
    text.append(f' 等权收益: {ew:+.2%}  超额: {exc:+.2%}')
    text.append(f' 加权收益: {pw:+.2%}  Spearman: {sp:.4f}')
text.append('')
text.append(f'基准收益: {bmk_total:+.2%}')
text.append(f'测试期: {len(dates_str)} 季')
text.append(f'训练期: 2020Q1-2021Q2 (6季)')
text.append(f'选股: 每期 Top {TOP_K}')

ax.text(0.05, 0.5, '\n'.join(text), fontsize=10, va='center', fontfamily=['SimSun', 'Microsoft YaHei'], linespacing=1.5)

plt.tight_layout()
fig_path = os.path.join(OUT_DIR, 'top30_overview.png')
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'\n回测总览图: {fig_path}')

# ── 各期 TOP30 详情图 ──
fig2, axes2 = plt.subplots(len(all_periods), 1, figsize=(14, 3*len(all_periods)))
if len(all_periods) == 1:
    axes2 = [axes2]
fig2.suptitle(f'各期 Top {TOP_K} 选股详情 (RF 模型)', fontsize=14, fontweight='bold')

for j, period in enumerate(all_periods):
    ax = axes2[j]
    codes = [str(c) for c in period['RandomForest_top_codes']]
    top_pred = period['RandomForest_score'][:TOP_K]
    # actual returns for top 30
    test = df[df['Date'] == pd.Timestamp(period['test_date'])]
    order = np.argsort(-np.array(top_pred))
    top_actual = test.iloc[order[:TOP_K]]['Next_Ret'].values

    y_pos = range(len(codes))
    ax.barh([y + 0.2 for y in y_pos], top_pred[:TOP_K], 0.35, label='预测收益', color='#378ADD', alpha=0.7)
    ax.barh(y_pos, top_actual, 0.35, label='实际收益', color='#D85A30', alpha=0.7)
    ax.set_yticks([y + 0.1 for y in y_pos])
    ax.set_yticklabels(codes, fontsize=8)
    ax.set_xlabel('收益率')
    ax.set_title(f'{period["test_date"]}  Top {TOP_K}  (基准: {period["benchmark"]:+.2%})')
    ax.axvline(x=period['benchmark'], color='gray', ls=':', lw=1, label='基准')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
detail_path = os.path.join(OUT_DIR, 'top30_details.png')
plt.savefig(detail_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'TOP30 详情: {detail_path}')

# ── 保存 JSON 结果 ──
out = {
    'summary': {
        'n_quarters': len(dates),
        'train_quarters': 6,
        'test_quarters': 4,
        'n_stocks': df['Code'].nunique(),
        'n_features': len(FEATURE_COLS),
        'top_k': TOP_K,
    },
    'models': {},
}
for mname in models:
    out['models'][mname] = {
        'total_return_ew': round(results_df[f'{mname}_ew'].sum(), 4),
        'total_return_pw': round(results_df[f'{mname}_pw'].sum(), 4),
        'avg_spearman': round(results_df[f'{mname}_spearman'].mean(), 4),
    }
out['benchmark_total_return'] = round(bmk_total, 4)
out['periods'] = [
    {
        'test_date': str(p['test_date']),
        'benchmark': round(p['benchmark'], 4),
        **{f'{m}_{wt}': round(p[f'{m}_{wt}'], 4)
           for m in models for wt in ['ew','pw']}
    }
    for p in all_periods
]

json_path = os.path.join(OUT_DIR, 'top30_result.json')
with open(json_path, 'w', encoding='utf-8') as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f'结果 JSON: {json_path}')
print('\n✅ Top 30 策略回测完成！')
