"""
从 top30_result.json + enhanced_result.json 生成综合 HTML 报告
明确标注数据期数（10 期季度数据：2020Q1-2022Q2）
"""
import json, os

OUT = r'C:\Users\DELL\WorkBuddy\2026-07-17-13-03-09'
top = json.load(open(os.path.join(OUT, 'top30_result.json'), encoding='utf-8'))
enh = json.load(open(os.path.join(OUT, 'enhanced_result.json'), encoding='utf-8'))

s = top['summary']
bmk = top['benchmark_total_return']
es = enh['scan_summary']
model_cn = {'LinearReg': '线性回归', 'LogisticReg': '逻辑回归',
            'DecisionTree': '决策树', 'RandomForest': '随机森林'}


def cls(v):
    return 'pos' if v > 0 else 'neg'


# ── Top30 基础 4 模型汇总 ──
base_rows = ''
for m, d in top['models'].items():
    ew, pw, sp = d['total_return_ew'], d['total_return_pw'], d['avg_spearman']
    base_rows += (
        f"<tr><td>{model_cn[m]} <span class='mut'>({m})</span></td>"
        f"<td class='{cls(ew)}'>{ew:+.2%}</td>"
        f"<td class='pos'>{ew - bmk:+.2%}</td>"
        f"<td class='{cls(pw)}'>{pw:+.2%}</td>"
        f"<td class='pos'>{pw - bmk:+.2%}</td>"
        f"<td>{sp:.4f}</td></tr>"
    )

# ── 各期明细 ──
period_rows = ''
for p in top['periods']:
    cells = f"<td>{p['test_date']}</td><td class='neg'>{p['benchmark']:+.2%}</td>"
    for m in ['LinearReg', 'LogisticReg', 'DecisionTree', 'RandomForest']:
        v = p[m + '_ew']
        cells += f"<td class='{cls(v)}'>{v:+.2%}</td>"
    period_rows += f"<tr>{cells}</tr>"

# ── 增强版参数扫描（按 Sharpe 降序）──
enh_sorted = sorted(enh['results'], key=lambda x: -x['sharpe'])
enh_rows = ''
for r in enh_sorted:
    tr, sh, dd, wr, sp = r['total_return'], r['sharpe'], r['max_drawdown'], r['win_rate'], r['avg_spearman']
    enh_rows += (
        f"<tr><td>{model_cn[r['model']]} <span class='mut'>({r['model']})</span></td>"
        f"<td>{r['topk']}</td>"
        f"<td class='{cls(tr)}'>{tr:+.2%}</td>"
        f"<td>{sh:.2f}</td>"
        f"<td class='neg'>{dd:+.2%}</td>"
        f"<td>{wr:.0%}</td>"
        f"<td>{sp:.3f}</td></tr>"
    )

css = """
* {margin:0;padding:0;box-sizing:border-box}
body {font-family:'Microsoft YaHei',sans-serif;background:#f5f6fa;color:#2d3436;padding:24px;line-height:1.6}
h1 {font-size:24px;margin-bottom:6px}
h2 {font-size:17px;color:#2d3436;margin:24px 0 12px;border-left:4px solid #6c5ce7;padding-left:10px}
.sub {color:#636e72;font-size:13px;margin-bottom:18px}
.card {background:#fff;border-radius:12px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06)}
.grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px;margin-bottom:20px}
.stat {text-align:center;background:#fff;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,0.06)}
.stat .v {font-size:24px;font-weight:700}
.stat .l {font-size:12px;color:#636e72;margin-top:4px}
.pos {color:#d63031} .neg {color:#00b894} .mut {color:#b2bec3;font-size:11px}
img {max-width:100%;border-radius:8px;margin:12px 0}
table {width:100%;border-collapse:collapse;font-size:13px}
th {background:#6c5ce7;color:#fff;padding:9px 12px}
td {padding:8px 12px;border-bottom:1px solid #f1f2f6;text-align:center}
tbody tr:nth-child(even) {background:#fafbff}
.note {font-size:12px;color:#636e72;margin-top:8px}
"""

body = f"""
<h1>机器学习选股策略报告 · Top30 季度轮动</h1>
<p class="sub">数据区间：<b>2020Q1 – 2022Q2（共 {s['n_quarters']} 个季度）</b> · {s['n_stocks']:,} 只股票 ·
{s['n_features']} 个基础财务因子 + {es['n_features_tech']} 个技术派生特征 = {es['n_features_total']} 维特征 ·
训练 {s['train_quarters']} 期 / 测试 {s['test_quarters']} 期 · 每期选 Top {s['top_k']}</p>

<div class="grid">
  <div class="stat"><div class="v">10 期</div><div class="l">数据季度数</div></div>
  <div class="stat"><div class="v">{s['n_stocks']:,}</div><div class="l">覆盖股票数</div></div>
  <div class="stat"><div class="v">{es['n_features_total']}</div><div class="l">总特征数 (基础{s['n_features']}+技术{es['n_features_tech']})</div></div>
  <div class="stat"><div class="v pos">{top['models']['RandomForest']['total_return_ew']:+.0%}</div><div class="l">RF 等权累计收益</div></div>
  <div class="stat"><div class="v" style="color:#6c5ce7">{enh['best_config']['sharpe']:.2f}</div><div class="l">增强最优 Sharpe (RF TopK={enh['best_config']['topk']})</div></div>
</div>

<div class="card">
  <h2>一、Top30 基础策略（4 模型 · 等权 vs 预测加权）</h2>
  <p class="note">收益为测试期（{s['test_quarters']} 个季度）等权累计；超额 = 收益 − 基准（基准累计 {bmk:+.2%}）。</p>
  <table><thead><tr><th>模型</th><th>等权累计</th><th>等权超额</th><th>加权累计</th><th>加权超额</th><th>Spearman</th></tr></thead>
  <tbody>{base_rows}</tbody></table>
</div>

<div class="card">
  <h2>二、各测试期收益明细（等权）</h2>
  <table><thead><tr><th>测试期</th><th>基准</th><th>线性回归</th><th>逻辑回归</th><th>决策树</th><th>随机森林</th></tr></thead>
  <tbody>{period_rows}</tbody></table>
</div>

<div class="card">
  <h2>三、增强版参数扫描（技术指标 + TopK=5/10/20/30/50，按 Sharpe 排序）</h2>
  <p class="note">基准累计 {es['benchmark_total_return']:+.2%} · 基准 Sharpe {es['benchmark_sharpe']:.2f}。最优配置：{enh['best_config']['model']} TopK={enh['best_config']['topk']}（收益 {enh['best_config']['total_return']:+.2%} / Sharpe {enh['best_config']['sharpe']:.2f}）。</p>
  <table><thead><tr><th>模型</th><th>TopK</th><th>累计收益</th><th>Sharpe</th><th>最大回撤</th><th>胜率</th><th>Spearman</th></tr></thead>
  <tbody>{enh_rows}</tbody></table>
</div>

<div class="card"><img src="top30_overview.png" alt="Top30 策略总览"></div>
<div class="card"><img src="enhanced_analysis.png" alt="增强分析图"></div>
"""

html = ("<!DOCTYPE html><html lang='zh-CN'><head><meta charset='UTF-8'>"
        "<title>Top30 选股策略报告（10期数据）</title>" + css + "</head><body>" + body + "</body></html>")

html_path = os.path.join(OUT, 'report_top30.html')
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print(f'✅ 报告已生成: {html_path}  ({len(html)} 字节)')
