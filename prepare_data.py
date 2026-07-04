# -*- coding: utf-8 -*-
"""Convert stock CSV files to compact JSON for web dashboard"""
import pandas as pd
import json
import os

DATA_DIR = 'C:/Users/DELL/Desktop/龙虾计划/选修备份文件_extracted/stock_data'
OUTPUT_DIR = 'C:/Users/DELL/Desktop/龙虾计划/选修备份文件_extracted/ai-quant/data'

os.makedirs(OUTPUT_DIR, exist_ok=True)

STOCKS = [
    {'file': '000756_新华制药_qfq.csv', 'name': '新华制药', 'code': '000756', 'market': 'A股'},
    {'file': '605507_国邦药业_qfq.csv', 'name': '国邦药业', 'code': '605507', 'market': 'A股'},
    {'file': '600079_ST人福_qfq.csv',   'name': 'ST人福',   'code': '600079', 'market': 'A股'},
    {'file': '00719_新华制药H_qfq.csv',  'name': '新华制药H', 'code': '00719', 'market': '港股'},
]

all_data = {}

for stock in STOCKS:
    filepath = os.path.join(DATA_DIR, stock['file'])
    df = pd.read_csv(filepath)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Compact format: arrays instead of objects
    dates = df['date'].tolist()
    ohlcv = []
    for _, row in df.iterrows():
        ohlcv.append([
            float(row['open']),
            float(row['close']),
            float(row['low']),
            float(row['high']),
            int(row['volume'])
        ])
    
    stock_data = {
        'name': stock['name'],
        'code': stock['code'],
        'market': stock['market'],
        'count': len(df),
        'dates': dates,
        'ohlcv': ohlcv,
    }
    
    # Save individual file
    outpath = os.path.join(OUTPUT_DIR, f"{stock['code']}.json")
    with open(outpath, 'w', encoding='utf-8') as f:
        json.dump(stock_data, f, ensure_ascii=False)
    print(f"Saved {outpath} ({len(dates)} records)")
    
    all_data[stock['code']] = stock_data

# Save combined file
combined_path = os.path.join(OUTPUT_DIR, 'all_stocks.json')
with open(combined_path, 'w', encoding='utf-8') as f:
    json.dump(all_data, f, ensure_ascii=False)
print(f"\nCombined file saved: {combined_path}")
print(f"File size: {os.path.getsize(combined_path) / 1024:.1f} KB")
