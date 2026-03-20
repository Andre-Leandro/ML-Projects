import pandas as pd
import numpy as np
import sys

train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

section = sys.argv[1] if len(sys.argv) > 1 else '1'

if section == '1':
    for col in ['region', 'primary_category', 'seller_tier', 'promo_tool', 'tool_type', 'cost_tier']:
        print(f"{col}: {train[col].nunique()} unique -> {list(train[col].unique())}")
elif section == '2':
    print("MISSING TRAIN:")
    m = train.isnull().sum()
    print(m[m > 0])
    print("\nMISSING TEST:")
    m2 = test.isnull().sum()
    print(m2[m2 > 0])
elif section == '3':
    q_stats = train.groupby('query_id')['is_relevant'].sum()
    print("Relevants per query distribution:")
    print(q_stats.value_counts().sort_index())
elif section == '4':
    all_ch = train['active_channels'].dropna().str.split('|').explode()
    print("Active channels:")
    print(all_ch.value_counts())
    print("\nRecent promo tools:")
    all_pt = train['recent_promo_tools'].dropna().str.split('|').explode()
    print(all_pt.value_counts())
elif section == '5':
    print("Promo tool relevance rates:")
    print(train.groupby('promo_tool')['is_relevant'].mean().sort_values(ascending=False))
    print("\nTool type relevance:")
    print(train.groupby('tool_type')['is_relevant'].mean().sort_values(ascending=False))
    print("\nSeller tier relevance:")
    print(train.groupby('seller_tier')['is_relevant'].mean().sort_values(ascending=False))
    print("\ntool_recently_used relevance:")
    print(train.groupby('tool_recently_used')['is_relevant'].mean())
elif section == '6':
    print("Date range train:", train['snapshot_date'].min(), "to", train['snapshot_date'].max())
    print("Date range test:", test['snapshot_date'].min(), "to", test['snapshot_date'].max())
    print("Train sellers:", train['seller_id'].nunique())
    print("Test sellers:", test['seller_id'].nunique())
    overlap = set(train['seller_id'].unique()) & set(test['seller_id'].unique())
    print("Overlapping sellers:", len(overlap))
    print("Numeric cols:", train.select_dtypes(include=[np.number]).columns.tolist())
