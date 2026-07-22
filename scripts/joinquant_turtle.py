"""
海龟交易法则 — 聚宽 (JoinQuant) v2.0
========================================
适用平台: https://www.joinquant.com

核心规则:
  - 入场: 收盘价 > 过去 N 日最高点 (唐奇安通道突破)
  - 加仓: 每上涨 0.5N 加仓一次, 最多 4 次
  - 止损: 入场价向下 2N
  - 出场: 收盘价 < 过去 M 日最低点
"""

# ============================================================
# 策略参数 (回测前可在聚宽UI中修改)
# ============================================================
g.entry_period = 20          # 入场突破周期
g.exit_period = 10           # 出场突破周期
g.atr_period = 20            # ATR 周期
g.risk_per_unit = 0.01       # 单 Unit 风险比例 (账户 1%)
g.max_units = 4              # 最大加仓次数
g.add_step = 0.5             # 加仓间隔 (0.5N)
g.stop_loss_n = 2.0          # 止损倍数 (2N)


def initialize(context):
    """策略初始化"""
    # ===== 选择回测标的 =====
    g.stock = '000756.XSHE'   # 新华制药
    # g.stock = '605507.XSHG' # 国邦医药
    # g.stock = '600079.XSHG' # ST人福

    # 手续费
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                             open_commission=0.0003,
                             close_commission=0.0003,
                             close_today_commission=0,
                             min_commission=5), type='stock')

    # 运行主逻辑 (每天)
    run_daily(main_logic, time='every_bar')

    # 状态变量
    g.position = 0
    g.entry_prices = []

    log.info(f"[初始化] 标的={g.stock} 策略参数: "
             f"入场={g.entry_period}日 出场={g.exit_period}日 "
             f"ATR={g.atr_period}日 止损={g.stop_loss_n}N")


def main_logic(context):
    """每日策略逻辑"""
    stock = g.stock

    # ================================================================
    # STEP 1: 获取数据 (保证数据量足够)
    # ================================================================
    need = max(g.entry_period, g.exit_period, g.atr_period) + 30
    df = attribute_history(stock, need, '1d',
                           ['close', 'high', 'low'],
                           skip_paused=True, fq='pre')

    if df is None or len(df) < need:
        return

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    dates = df.index

    # 今天的值
    today_close = close[-1]
    today_high = high[-1]
    today_low = low[-1]

    # ================================================================
    # STEP 2: 计算唐奇安通道 + ATR (全用昨天数据, 防未来函数)
    # ================================================================
    entry_high = 0.0
    exit_low = 0.0
    n_value = 0.0

    if len(close) >= g.entry_period + 1:
        # 昨日之前的 entry_period 日最高
        entry_high = max(high[-(g.entry_period+1):-1])
    if len(close) >= g.exit_period + 1:
        exit_low = min(low[-(g.exit_period+1):-1])

    # ATR: 用昨天及之前的数据计算
    if len(close) >= g.atr_period + 2:
        tr_list = []
        for i in range(-g.atr_period-1, 0):
            tr = max(high[i] - low[i],
                     abs(high[i] - close[i-1]),
                     abs(low[i] - close[i-1]))
            tr_list.append(tr)
        # 简单EMA
        alpha = 2.0 / (g.atr_period + 1)
        n_value = tr_list[0]
        for tr_val in tr_list[1:]:
            n_value = tr_val * alpha + n_value * (1 - alpha)
    else:
        return

    if n_value <= 0:
        return

    # ================================================================
    # STEP 3: 获取当前持仓
    # ================================================================
    position = context.portfolio.positions.get(stock)
    current_shares = position.total_amount if position else 0

    # ================================================================
    # STEP 4: 调试日志 (所有关键信息每天打印)
    # ================================================================
    if len(close) >= g.entry_period + 1:
        can_buy = today_close > entry_high
        can_sell = g.position > 0 and today_close < exit_low
        log.info(
            f"[信号] {dates[-1].strftime('%Y-%m-%d')} "
            f"close={today_close:.2f} "
            f"entry_high={entry_high:.2f} "
            f"exit_low={exit_low:.2f} "
            f"N={n_value:.3f} "
            f"持仓单位={g.position} "
            f"持仓股数={current_shares} "
            f"现金={context.portfolio.available_cash:.0f} "
            f"可以买入?={can_buy} "
            f"可以卖出?={can_sell}"
        )

    # ================================================================
    # STEP 5: 止损检查
    # ================================================================
    if g.position > 0 and len(g.entry_prices) > 0:
        last_entry = g.entry_prices[-1]
        stop_price = last_entry - g.stop_loss_n * n_value
        if today_low <= stop_price:
            order_target_value(stock, 0)
            log.info(f"[执行 - 止损] {dates[-1].strftime('%Y-%m-%d')} "
                     f"止损价={stop_price:.2f} 持仓={current_shares}")
            g.position = 0
            g.entry_prices = []
            return

    # ================================================================
    # STEP 6: 出场检查
    # ================================================================
    if g.position > 0 and current_shares > 0:
        if today_close < exit_low:
            order_target_value(stock, 0)
            log.info(f"[执行 - 出场] {dates[-1].strftime('%Y-%m-%d')} "
                     f"跌破{g.exit_period}日低点 {exit_low:.2f} "
                     f"持仓={current_shares}")
            g.position = 0
            g.entry_prices = []
            return

    # ================================================================
    # STEP 7: 入场/加仓
    # ================================================================
    should_buy = False
    buy_reason = ""

    if g.position == 0:
        # 首次入场
        if today_close > entry_high:
            should_buy = True
            buy_reason = f"突破{g.entry_period}日高点{entry_high:.2f}"
    elif g.position < g.max_units:
        # 加仓: 价格比上次入场价上涨 0.5N
        last_entry = g.entry_prices[-1]
        target = last_entry + g.add_step * n_value
        if today_close > target:
            should_buy = True
            buy_reason = f"加仓第{g.position+1}次 目标{target:.2f}"

    # ================================================================
    # STEP 8: 执行买入
    # ================================================================
    if should_buy:
        # 计算 1 Unit = 账户 1% 风险对应的股数
        unit_shares = int(context.portfolio.total_value * g.risk_per_unit / n_value)
        unit_shares = max(100, (unit_shares // 100) * 100)
        cost = today_close * unit_shares

        if context.portfolio.available_cash >= cost:
            order_value(stock, cost)
            g.position += 1
            g.entry_prices.append(today_close)
            log.info(
                f"[执行 - 买入] {dates[-1].strftime('%Y-%m-%d')} "
                f"价格={today_close:.2f} "
                f"股数={unit_shares} "
                f"金额={cost:.0f} "
                f"Unit={g.position}/{g.max_units} "
                f"买入原因={buy_reason}"
            )
        else:
            log.info(
                f"[跳过] 现金不足: 需要{cost:.0f} 可用{context.portfolio.available_cash:.0f}"
            )
