# -*- coding: utf-8 -*-
"""
股票数据增量更新脚本
功能：
1. 读取现有CSV文件，找到最后日期
2. 从腾讯财经API获取前复权(qfq)增量数据
3. 追加到CSV文件（去重）
4. 重新生成JSON文件供可视化页面使用
5. 支持本地运行和GitHub Actions自动运行

使用方法：
  python scripts/update_stock_data.py          # 增量更新
  python scripts/update_stock_data.py --full   # 全量重新下载
"""

import requests
import pandas as pd
import json
import os
import sys
import time
from datetime import datetime, timedelta

# === 配置 ===
STOCKS = [
    {"code": "000756", "name": "新华制药",  "market": "A股", "exchange": "sz", "start": "2018-01-01"},
    {"code": "605507", "name": "国邦药业",  "market": "A股", "exchange": "sh", "start": "2021-01-01"},
    {"code": "600079", "name": "ST人福",    "market": "A股", "exchange": "sh", "start": "2018-01-01"},
    {"code": "00719",  "name": "新华制药H", "market": "港股", "exchange": "hk", "start": "2018-01-01"},
    # === 第二批：医药制药板块 ===
    {"code": "300497", "name": "富祥药业",  "market": "A股", "exchange": "sz", "start": "2015-01-01"},
    {"code": "300583", "name": "赛托生物",  "market": "A股", "exchange": "sz", "start": "2017-01-01"},
    {"code": "300636", "name": "同和药业",  "market": "A股", "exchange": "sz", "start": "2017-01-01"},
    {"code": "603538", "name": "美诺华",    "market": "A股", "exchange": "sh", "start": "2017-01-01"},
    {"code": "002923", "name": "润都股份",  "market": "A股", "exchange": "sz", "start": "2018-01-01"},
]

# 路径（相对于repo根目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_DIR = os.path.dirname(SCRIPT_DIR)
CSV_DIR = os.path.join(REPO_DIR, "csv")
JSON_DIR = os.path.join(REPO_DIR, "data")

API_BASE = "http://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
ADJUST = "qfq"  # 前复权
MAX_COUNT = 640  # API每次最多返回640条
RETRY_TIMES = 3
RETRY_DELAY = 3  # 秒


def get_api_code(stock):
    """生成API所需的股票代码格式"""
    code = stock["code"]
    exch = stock["exchange"]
    if exch == "hk":
        return f"hk{code}"
    return f"{exch}{code}"


def fetch_kline(api_code, start_date, end_date, count=MAX_COUNT):
    """从腾讯API获取K线数据"""
    param = f"{api_code},day,{start_date},{end_date},{count},{ADJUST}"
    url = f"{API_BASE}?param={param}"

    for attempt in range(RETRY_TIMES):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            if "data" not in data or api_code not in data["data"]:
                print(f"  [警告] API返回数据为空: {api_code}")
                return []

            stock_data = data["data"][api_code]
            # qfq数据在 qfqday 键，不复权在 day 键
            kline_data = stock_data.get("qfqday") or stock_data.get("day", [])
            return kline_data
        except Exception as e:
            print(f"  [重试 {attempt+1}/{RETRY_TIMES}] 请求失败: {e}")
            if attempt < RETRY_TIMES - 1:
                time.sleep(RETRY_DELAY)
    return []


def fetch_full_history(stock):
    """全量下载股票历史数据（分段请求）"""
    api_code = get_api_code(stock)
    all_data = []
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = stock["start"]

    print(f"  全量下载 {stock['name']}({stock['code']}) 从 {start_date} 到 {end_date}")

    # 安全限制：最多迭代800次（对应约50万条记录，远超任何正常股票）
    max_iterations = 800
    iteration = 0

    while iteration < max_iterations:
        iteration += 1
        batch = fetch_kline(api_code, start_date, end_date, MAX_COUNT)
        if not batch:
            break

        all_data.extend(batch)

        # 如果返回不足MAX_COUNT条，说明已全部获取
        if len(batch) < MAX_COUNT:
            print(f"    获取 {len(batch)} 条（最后一批），累计 {len(all_data)} 条")
            break

        print(f"    获取 {len(batch)} 条，累计 {len(all_data)} 条")

        # 向前滚动end_date（使用最后一条数据的日期）
        last_date = batch[-1][0]
        # 如果最后日期早于start_date，说明已获取到足够早的数据
        if last_date <= start_date:
            print(f"    已到达起始日期 {start_date}，停止")
            break
        end_date = last_date
        time.sleep(0.5)
    else:
        print(f"  [警告] 达到最大迭代次数 {max_iterations}，停止下载")

    return all_data


def fetch_incremental(stock, last_date):
    """增量下载：从last_date之后获取新数据"""
    api_code = get_api_code(stock)
    today = datetime.now().strftime("%Y-%m-%d")

    # 从last_date的下一天开始
    dt = datetime.strptime(last_date, "%Y-%m-%d")
    start = (dt + timedelta(days=1)).strftime("%Y-%m-%d")

    if start > today:
        print(f"  {stock['name']} 数据已是最新 ({last_date})")
        return []

    print(f"  增量下载 {stock['name']}({stock['code']}) 从 {start} 到 {today}")
    batch = fetch_kline(api_code, start, today, MAX_COUNT)

    # 过滤掉已存在的日期（双重保险）
    new_data = [row for row in batch if row[0] > last_date]
    print(f"    新增 {len(new_data)} 条")
    return new_data


def kline_to_dataframe(kline_data, stock):
    """将API返回的K线数据转为DataFrame"""
    if not kline_data:
        return pd.DataFrame()

    rows = []
    for item in kline_data:
        # 格式: [date, open, close, high, low, volume]
        date = item[0]
        open_p = float(item[1]) if item[1] else 0
        close_p = float(item[2]) if item[2] else 0
        high_p = float(item[3]) if item[3] else 0
        low_p = float(item[4]) if item[4] else 0
        vol = float(item[5]) if len(item) > 5 and item[5] else 0
        rows.append({
            "date": date,
            "symbol": stock["code"],
            "name": stock["name"],
            "open": open_p,
            "close": close_p,
            "high": high_p,
            "low": low_p,
            "volume": vol,
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def update_csv(stock, force_full=False):
    """更新单只股票的CSV文件"""
    csv_path = os.path.join(CSV_DIR, f"{stock['code']}.csv")
    os.makedirs(CSV_DIR, exist_ok=True)

    existing_df = None
    if not force_full and os.path.exists(csv_path):
        try:
            existing_df = pd.read_csv(csv_path, encoding="utf-8-sig")
            existing_df = existing_df.sort_values("date").reset_index(drop=True)
        except Exception as e:
            print(f"  [警告] 读取CSV失败，将全量下载: {e}")
            existing_df = None

    if existing_df is not None and len(existing_df) > 0:
        last_date = existing_df["date"].iloc[-1]
        print(f"  现有数据: {len(existing_df)} 条，最后日期: {last_date}")

        new_data = fetch_incremental(stock, last_date)
        if not new_data:
            return existing_df

        new_df = kline_to_dataframe(new_data, stock)
        combined = pd.concat([existing_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["date"], keep="last")
        combined = combined.sort_values("date").reset_index(drop=True)
        print(f"  合并后: {len(combined)} 条")
    else:
        print(f"  无现有数据，全量下载...")
        kline_data = fetch_full_history(stock)
        combined = kline_to_dataframe(kline_data, stock)
        print(f"  全量下载: {len(combined)} 条")

    # 保存CSV
    combined.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"  已保存: {csv_path}")
    return combined


def generate_json(stocks_data):
    """从CSV数据生成JSON文件"""
    os.makedirs(JSON_DIR, exist_ok=True)
    all_data = {}

    for stock in STOCKS:
        df = stocks_data[stock["code"]]
        if df is None or len(df) == 0:
            continue

        dates = df["date"].tolist()
        ohlcv = []
        for _, row in df.iterrows():
            ohlcv.append([
                round(float(row["open"]), 3),
                round(float(row["close"]), 3),
                round(float(row["low"]), 3),
                round(float(row["high"]), 3),
                int(row["volume"]),
            ])

        stock_data = {
            "name": stock["name"],
            "code": stock["code"],
            "market": stock["market"],
            "count": len(df),
            "dates": dates,
            "ohlcv": ohlcv,
            "updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        outpath = os.path.join(JSON_DIR, f"{stock['code']}.json")
        with open(outpath, "w", encoding="utf-8") as f:
            json.dump(stock_data, f, ensure_ascii=False)
        print(f"  JSON已保存: {outpath} ({len(dates)} 条)")

        all_data[stock["code"]] = stock_data

    # 合并文件
    combined_path = os.path.join(JSON_DIR, "all_stocks.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, ensure_ascii=False)
    size_kb = os.path.getsize(combined_path) / 1024
    print(f"  合并文件: {combined_path} ({size_kb:.1f} KB)")


def main():
    force_full = "--full" in sys.argv

    print("=" * 60)
    print(f"股票数据增量更新 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"模式: {'全量下载' if force_full else '增量更新'}")
    print(f"复权方式: 前复权(qfq)")
    print("=" * 60)

    stocks_data = {}
    total_new = 0

    for stock in STOCKS:
        print(f"\n[{stock['name']} ({stock['code']})]")
        df = update_csv(stock, force_full=force_full)
        stocks_data[stock["code"]] = df

    print("\n" + "=" * 60)
    print("生成JSON文件...")
    generate_json(stocks_data)

    print("\n" + "=" * 60)
    print("更新完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
