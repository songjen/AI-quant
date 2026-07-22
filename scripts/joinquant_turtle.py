"""
海龟交易法则 — 聚宽 (JoinQuant) 兼容版本
=========================================
适用平台: https://www.joinquant.com
使用方法: 将此代码完整粘贴到聚宽的策略编辑器中即可运行回测

核心规则:
  - 入场: 价格突破过去 N 日最高点 (唐奇安通道上轨)
  - 加仓: 每上涨 0.5N 加仓一次, 最多 4 次 (含初始)
  - 止损: 最后一次入场价向下 2N
  - 出场: 价格跌破过去 M 日最低点 (唐奇安通道下轨)

仓位计算:
  - N = ATR(20)
  - 1 Unit = 总资产的 1% / (N × 收盘价)
  - 不超过最大仓位限制
"""

# ============================================================
# 策略参数 (可在聚宽 UI 中调节)
# ============================================================
g.entry_period = 20          # 唐奇安入场周期
g.exit_period = 10           # 唐奇安出场周期
g.atr_period = 20            # ATR 计算周期
g.risk_per_unit = 0.01       # 每 Unit 风险比例 (账户 1%)
g.max_units = 4              # 最大加仓次数 (含首次)
g.add_step = 0.5             # 加仓间隔 (0.5N)
g.stop_loss_n = 2.0          # 止损倍数 (2N)
g.use_system2 = False        # False=System1(20日), True=System2(55日)


# ============================================================
# 初始化
# ============================================================
def initialize(context):
    """策略初始化"""
    # 标的
    g.stock = '000756.XSHE'  # 新华制药 (可修改)
    # g.stock = '605507.XSHG'  # 国邦医药
    # g.stock = '600079.XSHG'  # ST人福

    set_universe(g.stock)

    # 手续费: 万分之三
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001,
                              open_commission=0.0003,
                              close_commission=0.0003,
                              close_today_commission=0,
                              min_commission=5), type='stock')

    # 运行主逻辑
    run_daily(main_logic, time='every_bar')

    # 状态变量
    g.position = 0            # 当前持仓 Unit 数
    g.shares = 0              # 持仓股数
    g.entry_prices = []       # 各次入场价格
    g.avg_cost = 0.0          # 持仓均价
    g.last_signal_date = None # 上一个信号日期 (防重复)


def main_logic(context):
    """每日执行的主逻辑"""
    stock = g.stock
    current_date = context.current_dt.date()

    # --- 第1步: 获取数据 ---
    # 需要足够长的数据窗口
    max_period = max(g.entry_period, g.exit_period, g.atr_period) + 30
    df = attribute_history(stock, max_period, '1d',
                           ['open', 'close', 'high', 'low', 'volume', 'money'],
                           skip_paused=True, fq='pre')
    if df is None or len(df) < max_period:
        return

    # --- 第2步: 计算指标 (全部 shift(1) 防未来函数) ---
    close = df['close']
    high = df['high']
    low = df['low']

    # 唐奇安通道 — 用昨日数据判断今日突破
    entry_high = high.rolling(g.entry_period).max().shift(1)
    entry_low = low.rolling(g.entry_period).min().shift(1)
    exit_low = low.rolling(g.exit_period).min().shift(1)

    # ATR 计算 (也用昨日值)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    n_value = tr.ewm(span=g.atr_period, adjust=False).mean().shift(1)

    # 当前值 (最新)
    cur_close = close.iloc[-1]
    cur_high = high.iloc[-1]
    cur_low = low.iloc[-1]
    cur_entry_high = entry_high.iloc[-1]
    cur_exit_low = exit_low.iloc[-1]
    cur_n = n_value.iloc[-1]

    if pd.isna(cur_n) or cur_n <= 0:
        return

    # --- 第3步: 检查持仓 (用 .get 避免 WARNING) ---
    position = context.portfolio.positions.get(stock)
    current_shares = position.total_amount if position else 0
    g.shares = current_shares

    # --- 第4步: 计算 1 Unit 股数 ---
    total_value = context.portfolio.total_value
    one_unit_value = total_value * g.risk_per_unit
    one_unit_shares = int(one_unit_value / cur_n)
    one_unit_shares = max(100, (one_unit_shares // 100) * 100)

    # --- 调试日志: 打印关键信号值 (前 60 天) ---
    if context.current_dt.date() < pd.Timestamp('2024-09-01').date():
        log.info(f"[DEBUG] date={current_date} close={cur_close:.2f} "
                 f"entry_high={cur_entry_high:.2f} exit_low={cur_exit_low:.2f} "
                 f"N={cur_n:.2f} position={g.position} "
                 f"shares={current_shares} buy?={cur_close > cur_entry_high}")

    # --- 第5步: 止损检查 ---
    if g.position > 0 and len(g.entry_prices) > 0:
        last_entry = g.entry_prices[-1]
        stop_price = last_entry - g.stop_loss_n * cur_n
        if cur_low <= stop_price and current_shares > 0:
            # 触发止损
            order_target_value(stock, 0)
            g.position = 0
            g.entry_prices = []
            g.avg_cost = 0.0
            log.info(f"[止损] {current_date} 止损价 {stop_price:.2f} 平仓")
            return

    # --- 第6步: 出场检查 (跌破 N 日低点) ---
    if g.position > 0 and current_shares > 0:
        if cur_low <= cur_exit_low:
            order_target_value(stock, 0)
            g.position = 0
            g.entry_prices = []
            g.avg_cost = 0.0
            log.info(f"[出场] {current_date} 跌破{g.exit_period}日低点 {cur_exit_low:.2f}")
            return

    # --- 第7步: 入场/加仓 ---
    buy_price = None
    buy_reason = ""

    if g.position == 0:
        # 首次入场: 收盘价突破 N 日高点
        if cur_close > cur_entry_high:
            buy_price = cur_close
            buy_reason = f"突破{g.entry_period}日高点 {cur_entry_high:.2f}"
    elif g.position < g.max_units:
        # 加仓: 价格较上次入场价上涨 0.5N
        last_entry = g.entry_prices[-1]
        target_price = last_entry + g.add_step * cur_n
        if cur_close > target_price:
            buy_price = cur_close
            buy_reason = f"加仓(第{g.position+1}次) {target_price:.2f}"

    # --- 第8步: 执行买入 ---
    if buy_price is not None:
        # 计算买入股数
        buy_shares = one_unit_shares

        # 检查现金是否足够
        cost = buy_price * buy_shares
        if context.portfolio.available_cash >= cost:
            order_value(stock, cost)
            g.position += 1
            g.entry_prices.append(buy_price)

            # 更新均价
            total_cost = sum(p * buy_shares for p in g.entry_prices)
            g.avg_cost = total_cost / (g.position * buy_shares)

            # 计算止损价
            stop = g.entry_prices[-1] - g.stop_loss_n * cur_n
            log.info(f"[买入] {current_date} {buy_reason} | "
                     f"价格: {buy_price:.2f} | "
                     f"Unit: {g.position}/{g.max_units} | "
                     f"止损: {stop:.2f} | 数量 {buy_shares}")
    elif current_shares == 0:
        # 重置状态 (如因初始化问题)
        g.position = 0
        g.entry_prices = []


# ============================================================
# 辅助函数 (用于绩效报告)
# ============================================================
def calc_atr(high, low, close, period=20):
    """计算 ATR"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


# ============================================================
# 实现 System 2: 55 日突破 (可选)
# ============================================================
# 将 g.use_system2 设为 True 后, 入场改为 55 日突破
# System1: 入场 20日 / 出场 10日 (趋势敏感)
# System2: 入场 55日 / 出场 20日 (趋势迟钝, 胜率高)
