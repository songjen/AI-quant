"""
海龟交易法则 — 完全对齐JS版 v4.2
===================================
修复:
- order_value + 佣金双重扣除问题，改用 order() 精确控制股数
- 止损/出场后不 return，对齐 JS 同一根 K 线可再入场
- ATR 计算改为从策略起始日逐步 Wilder 平滑，与 JS 全历史口径一致
- 卖出费率对齐 JS 的 0.08%（印花税+佣金合计）
- 标的改为 000756.XSHE 与当前调试日志一致
"""

g.entry_period = 20
g.exit_period = 10
g.atr_period = 20
g.risk_per_unit = 0.01
g.max_units = 4
g.add_step = 0.5
g.stop_loss_n = 2.0
g.slippage = 0.001
g.buy_cost = 0.0003
g.sell_cost = 0.0008


def initialize(context):
    g.stock = '000756.XSHE'

    # 对齐 JS 版 sellCost=0.0008：卖出总成本 = 印花税 + 佣金 = 0.0005 + 0.0003
    set_order_cost(OrderCost(open_tax=0, close_tax=0.0005,
                             open_commission=0.0003,
                             close_commission=0.0003,
                             close_today_commission=0,
                             min_commission=5), type='stock')

    run_daily(main_logic, time='every_bar')

    g.units = []
    g.position = 0

    # 记录策略起始日，用于后续拉取全历史数据（对齐 JS 从数据起点算 ATR）
    g.start_date = context.run_params.start_date

    log.info(f"[初始化] 标的={g.stock}")


def main_logic(context):
    stock = g.stock

    # === 拉取从策略起始日到昨日的全历史 ===
    # JS 版 turtle.html 从 2018-02-27 的数据起点开始算 Wilder ATR；
    # 聚宽版原来只取最近 50 根，导致 ATR 每天被重置，全历史结果与 JS 严重偏离。
    # 这里改为拉取策略开始至今的全部历史，确保 ATR 与唐奇安通道和 JS 同口径。
    hist_end = context.previous_date
    hist_start = g.start_date
    df_hist = get_price(stock, start_date=hist_start, end_date=hist_end,
                        frequency='1d', fields=['close', 'high', 'low', 'open'],
                        skip_paused=True, fq='pre')
    if df_hist is None or len(df_hist) < max(g.entry_period, g.exit_period, g.atr_period) + 2:
        return

    close = df_hist['close'].values
    high = df_hist['high'].values
    low = df_hist['low'].values
    open_p = df_hist['open'].values
    dates = df_hist.index

    # 今天的开盘价用于计算理论入场价（聚宽 every_bar 在开盘执行，成交价接近 today_open）
    today_open = get_current_data()[stock].day_open
    # 昨日收盘价/最高/最低用于判断离场与突破（对齐 JS 用上一根 bar 收盘后决策）
    today_close = close[-1]
    today_high = high[-1]
    today_low = low[-1]

    # === 唐奇安通道 ===
    entry_high = 0.0
    exit_low = 0.0
    if len(close) >= g.entry_period + 1:
        entry_high = max(high[-(g.entry_period+1):-1])
    if len(close) >= g.exit_period + 1:
        exit_low = min(low[-(g.exit_period+1):-1])

    # === ATR — Wilder平滑 (完全对齐JS) ===
    # JS 版：从 i=1 开始计算 TR，ATR[period]=mean(TR[1..period])，之后 Wilder 平滑到当前
    n_value = 0.0
    if len(close) >= g.atr_period + 2:
        tr_list = []
        for i in range(1, len(close)):
            tr = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            tr_list.append(tr)
        if len(tr_list) < g.atr_period + 1:
            return
        n_value = sum(tr_list[:g.atr_period]) / g.atr_period
        for tr_val in tr_list[g.atr_period:]:
            n_value = (n_value * (g.atr_period - 1) + tr_val) / g.atr_period
    else:
        return

    if n_value <= 0:
        return

    # === 持仓同步 ===
    # 用 positions.keys() 判断持仓，避免空仓日访问 positions 触发聚宽兼容 WARNING
    positions = context.portfolio.positions
    held_keys = list(positions.keys())
    pos = positions[stock] if stock in held_keys else None
    current_shares = pos.total_amount if (pos and pos.total_amount > 0) else 0
    if current_shares == 0:
        g.units = []
        g.position = 0

    # === 止损检查 (盘中最低价<=止损价) ===
    if g.position > 0 and len(g.units) > 0:
        last_unit = g.units[-1]
        current_stop = last_unit['stop_price']
        if today_low <= current_stop:
            order_target_value(stock, 0)
            log.info(f"[止损] {dates[-1].strftime('%Y-%m-%d')} "
                     f"触发价={current_stop:.2f}")
            g.units = []
            g.position = 0
            # 不 return! JS版在止损后同一根K线继续检查入场

    # === 出场检查 (收盘价<离场下轨) ===
    if g.position > 0 and current_shares > 0:
        if today_close < exit_low:
            order_target_value(stock, 0)
            log.info(f"[出场] {dates[-1].strftime('%Y-%m-%d')} "
                     f"跌破{g.exit_period}日低点{exit_low:.2f}")
            g.units = []
            g.position = 0
            # 不 return! JS版在出场后同一根K线继续检查入场

    # === 入场/加仓 ===
    should_buy = False
    buy_reason = ""
    entry_price = 0.0
    buy_shares = 0

    if g.position == 0 and len(g.units) == 0:
        if today_high > entry_high:
            should_buy = True
            buy_reason = f"突破{g.entry_period}日高点{entry_high:.2f}"
            entry_price = max(entry_high, today_open) * (1 + g.slippage)
    elif len(g.units) < g.max_units:
        last_unit = g.units[-1]
        addon_price = last_unit['entry_price'] + g.add_step * n_value
        if today_high >= addon_price:
            should_buy = True
            buy_reason = f"加仓第{len(g.units)+1}次"
            entry_price = addon_price * (1 + g.slippage)

    # === 执行买入 ===
    if should_buy and entry_price > 0:
        cash = context.portfolio.available_cash
        risk_amount = cash * g.risk_per_unit
        stop_distance = g.stop_loss_n * n_value
        if stop_distance > 0:
            buy_shares = int(risk_amount / stop_distance / 100) * 100
            if buy_shares < 100:
                buy_shares = 100

            # 只用本金买，佣金由JoinQuant自动处理
            principal = buy_shares * entry_price
            if cash >= principal:
                order(stock, buy_shares)
                stop_price = entry_price - stop_distance
                g.position += buy_shares
                g.units.append({
                    'shares': buy_shares,
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                })
                log.info(f"[买入] {dates[-1].strftime('%Y-%m-%d')} "
                         f"价={entry_price:.2f} 量={buy_shares} "
                         f"单元={len(g.units)}/{g.max_units} "
                         f"止损={stop_price:.2f}")
            else:
                log.info(f"[跳过] 需{principal:.0f} 有{cash:.0f}")
