"""
海龟交易法则 — 聚宽版 (JS版对齐) v3.0
========================================
完全对齐 https://songjen.github.io/AI-quant/turtle.html 的JS版本逻辑

核心差异(对比v2):
  1. 入场: high[t] > 唐奇安上轨 (JS原版, 用最高价)
  2. 入场价: max(突破价, 开盘价) × (1+滑点)
  3. ATR: Wilder平滑 (alpha=1/period, 不是EMA)
  4. 仓位: 现金 × 风险比例 / (止损倍数 × N)  (JS原版公式)
  5. 加仓: high[t] > 上次入场价 + add_step × N (用最高价)
  6. 止损: 按最新单元的止损价 (不是按均价)
"""

# ============================================================
# 策略参数
# ============================================================
g.entry_period = 20          # 入场突破周期
g.exit_period = 10           # 出场突破周期
g.atr_period = 20            # ATR 周期
g.risk_per_unit = 0.01       # 单 Unit 风险比例 (账户 1%)
g.max_units = 4              # 最大加仓次数
g.add_step = 0.5             # 加仓间隔 (0.5N)
g.stop_loss_n = 2.0          # 止损倍数 (2N)
g.slippage = 0.001           # 滑点 0.1%
g.buy_cost = 0.0003          # 买入费率 万3
g.sell_cost = 0.0008         # 卖出费率 (含印花税) 万8


def initialize(context):
    """策略初始化"""
    g.stock = '605507.XSHG'  # 国邦医药
    # g.stock = '000756.XSHE' # 新华制药
    # g.stock = '600079.XSHG' # ST人福

    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0003,
                             close_commission=0.0003,
                             close_today_commission=0,
                             min_commission=5), type='stock')

    run_daily(main_logic, time='every_bar')

    # 状态变量 — 使用数组跟踪每一单元的详细信息
    g.units = []        # [{shares, entry_price, stop_price, entry_n, entry_idx}, ...]
    g.position = 0      # 持仓总股数
    g.equity_peak = context.portfolio.total_value  # 用于记录峰值

    log.info(f"[初始化] 标的={g.stock} 对齐JS版海龟策略")


def main_logic(context):
    """每日策略逻辑"""
    stock = g.stock

    # ================================================================
    # STEP 1: 获取数据
    # ================================================================
    need = max(g.entry_period, g.exit_period, g.atr_period) + 30
    df = attribute_history(stock, need, '1d',
                           ['close', 'high', 'low', 'open'],
                           skip_paused=True, fq='pre')

    if df is None or len(df) < need:
        return

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    open_p = df['open'].values
    dates = df.index

    today_open = open_p[-1]
    today_close = close[-1]
    today_high = high[-1]
    today_low = low[-1]

    # ================================================================
    # STEP 2: 计算指标 (对齐JS版)
    # ================================================================
    # 唐奇安通道 — 取过去N日(不含今日)的最高/最低
    entry_high = 0.0
    exit_low = 0.0
    if len(close) >= g.entry_period + 1:
        entry_high = max(high[-(g.entry_period+1):-1])
    if len(close) >= g.exit_period + 1:
        exit_low = min(low[-(g.exit_period+1):-1])

    # ATR — Wilder平滑 (JS版算法)
    n_value = 0.0
    if len(close) >= g.atr_period + 2:
        tr_list = []
        for i in range(-g.atr_period-1, 0):
            tr = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            tr_list.append(tr)
        # Wilder平滑: first = SMA, then = (prev*(p-1)+TR)/p
        n_value = sum(tr_list[:g.atr_period]) / g.atr_period
        for tr_val in tr_list[g.atr_period:]:
            n_value = (n_value * (g.atr_period - 1) + tr_val) / g.atr_period
    else:
        return

    if n_value <= 0:
        return

    # ================================================================
    # STEP 3: 获取持仓
    # ================================================================
    position = context.portfolio.positions.get(stock)
    current_shares = position.total_amount if position else 0

    # 如果聚宽持仓为0但我们的状态还有，说明外部平仓了，重置
    if current_shares == 0:
        g.units = []
        g.position = 0

    # ================================================================
    # STEP 4: 止损检查 (优先级最高) — 用最新单元的止损价
    # ================================================================
    if g.position > 0 and len(g.units) > 0:
        last_unit = g.units[-1]
        current_stop = last_unit['stop_price']
        if today_low <= current_stop:
            # 按止损价卖出 (滑点扣减)
            sell_price = current_stop * (1 - g.slippage)
            order_target_value(stock, 0)
            log.info(f"[执行-止损] {dates[-1].strftime('%Y-%m-%d')} "
                     f"止损价={current_stop:.2f} 成交价={sell_price:.2f} "
                     f"持仓={current_shares} 单元数={len(g.units)}")
            g.units = []
            g.position = 0
            return

    # ================================================================
    # STEP 5: 出场检查 — 收盘价跌破唐奇安下轨
    # ================================================================
    if g.position > 0 and current_shares > 0:
        if today_close < exit_low:
            sell_price = today_close * (1 - g.slippage)
            order_target_value(stock, 0)
            log.info(f"[执行-出场] {dates[-1].strftime('%Y-%m-%d')} "
                     f"跌破{g.exit_period}日低点 {exit_low:.2f} "
                     f"成交价={sell_price:.2f} 持仓={current_shares}")
            g.units = []
            g.position = 0
            return

    # ================================================================
    # STEP 6: 入场/加仓 (用最高价判断 — 对齐JS版)
    # ================================================================
    should_buy = False
    buy_reason = ""
    entry_price = 0.0
    buy_shares = 0

    if g.position == 0 and len(g.units) == 0:
        # 首次入场: 最高价突破唐奇安上轨
        if today_high > entry_high:
            should_buy = True
            buy_reason = f"突破{g.entry_period}日高点{entry_high:.2f}"
            # 入场价 = max(突破价, 开盘价) × (1+滑点) — 对齐JS版
            entry_price = max(entry_high, today_open) * (1 + g.slippage)
    elif len(g.units) < g.max_units:
        # 加仓: 最高价 > 上次入场价 + 0.5N (对齐JS版)
        last_unit = g.units[-1]
        addon_price = last_unit['entry_price'] + g.add_step * n_value
        if today_high >= addon_price:
            should_buy = True
            buy_reason = f"加仓第{len(g.units)+1}次 目标{addon_price:.2f}"
            entry_price = addon_price * (1 + g.slippage)

    # ================================================================
    # STEP 7: 计算仓位并执行买入 (对齐JS版公式)
    # ================================================================
    if should_buy and entry_price > 0:
        # JS版仓位公式: shares = cash × riskPct / (stopMult × N)
        cash = context.portfolio.available_cash
        risk_amount = cash * g.risk_per_unit
        stop_distance = g.stop_loss_n * n_value
        if stop_distance > 0:
            buy_shares = int(risk_amount / stop_distance / 100) * 100
            if buy_shares < 100:
                buy_shares = 100

            cost = buy_shares * entry_price
            if cash >= cost and buy_shares > 0:
                order_value(stock, cost)
                g.position += buy_shares
                # 记录本次单元
                stop_price = entry_price - stop_distance
                g.units.append({
                    'shares': buy_shares,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'entry_n': n_value,
                })
                log.info(
                    f"[执行-买入] {dates[-1].strftime('%Y-%m-%d')} "
                    f"价格={entry_price:.2f} 股数={buy_shares} "
                    f"金额={cost:.0f} 单元={len(g.units)}/{g.max_units} "
                    f"止损={stop_price:.2f} ({g.stop_loss_n}N) "
                    f"原因={buy_reason}"
                )
            else:
                log.info(f"[跳过] 现金不足: 需要{cost:.0f} 可用{cash:.0f}")
