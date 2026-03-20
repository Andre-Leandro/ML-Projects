"""
Marketplace Promotion Recommendation — V3 Final Solution
==========================================================
LightGBM LambdaRank + Binary Ensemble
Expanding-Window Temporal Aggregates + Feature Engineering
Metric: NDCG@3
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('dataset/train.csv')
test = pd.read_csv('dataset/test.csv')

target = train['is_relevant'].values
train['snapshot_date'] = pd.to_datetime(train['snapshot_date'])
test['snapshot_date'] = pd.to_datetime(test['snapshot_date'])

# ============================================================
# 2. Expanding-Window Temporal Aggregates (Vectorized)
# ============================================================
print("Computing expanding-window aggregates (vectorized)...")

train = train.sort_values('snapshot_date').reset_index(drop=True)
weeks = sorted(train['snapshot_date'].unique())

def compute_expanding_agg(df, group_cols, target_col='is_relevant', prior=0.25, smooth=5):
    """Vectorized expanding-window target aggregation."""
    df = df.sort_values('snapshot_date').reset_index(drop=True)
    weeks = sorted(df['snapshot_date'].unique())
    
    col_name = '_'.join(group_cols) + '_exp'
    col_name_n = col_name + '_n'
    result = np.full(len(df), prior)
    result_n = np.zeros(len(df))
    
    cum_sum = {}
    cum_count = {}
    
    for week in weeks:
        mask = df['snapshot_date'] == week
        week_idx = np.where(mask)[0]
        week_data = df.loc[mask]
        
        # Assign rates from accumulated past data
        keys = week_data[group_cols].apply(tuple, axis=1)
        for i, key in zip(week_idx, keys):
            if key in cum_count:
                s, c = cum_sum[key], cum_count[key]
                result[i] = (s + prior * smooth) / (c + smooth)
                result_n[i] = c
        
        # Update accumulators
        for i, key in zip(week_idx, keys):
            if key not in cum_sum:
                cum_sum[key] = 0
                cum_count[key] = 0
            cum_sum[key] += df.loc[i, target_col]
            cum_count[key] += 1
    
    df[col_name] = result
    df[col_name_n] = result_n
    return df

# Key expanding aggregates
for gcols in [['seller_id', 'promo_tool'], ['seller_id'], 
              ['primary_category', 'promo_tool'], ['region', 'promo_tool'],
              ['seller_tier', 'promo_tool']]:
    print(f"  {gcols}...")
    train = compute_expanding_agg(train, gcols)

# For test: use all train data
def full_agg_for_test(train_df, test_df, group_cols, prior=0.25, smooth=5):
    col_name = '_'.join(group_cols) + '_exp'
    col_name_n = col_name + '_n'
    
    agg = train_df.groupby(group_cols)['is_relevant'].agg(['sum', 'count']).reset_index()
    agg[col_name] = (agg['sum'] + prior * smooth) / (agg['count'] + smooth)
    agg[col_name_n] = agg['count']
    agg = agg.drop(columns=['sum', 'count'])
    
    test_df = test_df.merge(agg, on=group_cols, how='left')
    test_df[col_name] = test_df[col_name].fillna(prior)
    test_df[col_name_n] = test_df[col_name_n].fillna(0)
    return test_df

for gcols in [['seller_id', 'promo_tool'], ['seller_id'],
              ['primary_category', 'promo_tool'], ['region', 'promo_tool'],
              ['seller_tier', 'promo_tool']]:
    test = full_agg_for_test(train, test, gcols)

# ============================================================
# 3. Basic Feature Engineering
# ============================================================
print("Engineering features...")

def add_features(df):
    df = df.copy()
    
    df['day_of_year'] = df['snapshot_date'].dt.dayofyear
    df['week_of_month'] = df['snapshot_date'].dt.day // 7 + 1
    
    # Multi-hot channels
    for ch in ['search', 'ads', 'social', 'affiliate', 'live']:
        df[f'ch_{ch}'] = df['active_channels'].fillna('').str.contains(ch).astype(int)
    df['n_channels'] = df[[f'ch_{ch}' for ch in ['search', 'ads', 'social', 'affiliate', 'live']]].sum(axis=1)
    
    # Multi-hot recent tools
    tools = ['homepage_feature', 'sponsored_search_boost', 'bundle_builder',
             'flash_sale_slot', 'loyalty_points_multiplier', 'coupon_pack',
             'free_shipping_boost', 'cashback_offer']
    for pt in tools:
        df[f'rpt_{pt}'] = df['recent_promo_tools'].fillna('').str.contains(pt).astype(int)
    df['n_recent_tools'] = df[[f'rpt_{pt}' for pt in tools]].sum(axis=1)
    
    # Tool-in-recent
    def check_match(row):
        if pd.isna(row['recent_promo_tools']):
            return 0
        return 1 if row['promo_tool'] in row['recent_promo_tools'].split('|') else 0
    df['tool_in_recent'] = df.apply(check_match, axis=1)
    
    # Interactions
    df['xborder_x_fit'] = df['is_cross_border'] * df['cross_border_fit']
    df['new_x_fit'] = (df['seller_tier'] == 'new').astype(int) * df['new_seller_fit']
    df['holiday_x_seasonal'] = df['holiday_campaign'] * df['seasonal_fit']
    df['monthend_x_discount'] = df['month_end_push'] * df['discount_depth']
    df['fulfill_x_inv'] = df['uses_fulfillment_service'] * df['inventory_synergy']
    df['loyalty_x_repeat'] = df['loyalty_synergy'] * df['repeat_buyer_rate']
    df['vis_x_search'] = df['visibility_boost'] * df['search_visibility_score']
    df['discount_x_margin'] = df['discount_depth'] * df['margin_rate']
    df['penalty_x_margin'] = df['margin_penalty'] * df['margin_rate']
    df['cashback_x_budget'] = df['cross_border_fit'] * df['cashback_budget_score']
    df['vis_x_conversion'] = df['visibility_boost'] * df['conversion_rate']
    df['inv_x_stockout'] = df['inventory_synergy'] * (1 - df['stockout_rate_30d'])
    df['inv_x_fill'] = df['inventory_synergy'] * df['inventory_fill_rate']
    df['fatigue_x_discount'] = df['promotion_fatigue_30d'] * df['discount_depth']
    df['readiness_x_vis'] = df['marketing_readiness_score'] * df['visibility_boost']
    df['new_fit_x_tenure'] = df['new_seller_fit'] / (df['seller_tenure_days'] + 1) * 100
    df['seasonal_x_quarter'] = df['seasonal_fit'] * df['quarter']
    df['loyalty_x_rating'] = df['loyalty_synergy'] * df['seller_rating']
    df['vis_x_gmv'] = df['visibility_boost'] * np.log1p(df['gmv_30d'])
    
    # Seller performance
    df['gmv_per_order'] = df['gmv_30d'] / (df['orders_30d'] + 1)
    df['ad_efficiency'] = df['gmv_30d'] / (df['ad_spend_30d'] + 1)
    df['view_to_order'] = df['orders_30d'] / (df['listing_views_30d'] + 1)
    df['readiness_minus_fatigue'] = df['marketing_readiness_score'] - df['promotion_fatigue_30d']
    df['fill_minus_stockout'] = df['inventory_fill_rate'] - df['stockout_rate_30d']
    df['ad_per_order'] = df['ad_spend_30d'] / (df['orders_30d'] + 1)
    df['margin_gmv'] = df['margin_rate'] * df['gmv_30d']
    
    # Tool composites
    df['tool_benefit'] = df['visibility_boost'] + df['inventory_synergy'] + df['loyalty_synergy'] - df['margin_penalty']
    df['tool_cost_benefit'] = (df['visibility_boost'] + df['inventory_synergy']) / (df['margin_penalty'] + 0.1)
    
    # Label encode
    for col in ['region', 'primary_category', 'seller_tier', 'promo_tool', 'tool_type', 'cost_tier']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    
    # Tier dummies
    for t in ['premium', 'established', 'growth', 'new']:
        df[f'is_{t}'] = (df['seller_tier'] == t).astype(int)
    for tt in ['visibility', 'bundle', 'event', 'retention', 'shipping', 'discount']:
        df[f'tt_{tt}'] = (df['tool_type'] == tt).astype(int)
    
    return df

train = add_features(train)
test = add_features(test)

# ============================================================
# 4. Target Encoding (K-Fold)
# ============================================================
print("Target encoding...")

def kfold_te(train_df, test_df, col, n_splits=5, smoothing=20):
    gm = train_df['is_relevant'].mean()
    enc = np.full(len(train_df), gm)
    kf = GroupKFold(n_splits=n_splits)
    for tri, vai in kf.split(train_df, train_df['is_relevant'], train_df['query_id']):
        agg = train_df.iloc[tri].groupby(col)['is_relevant'].agg(['sum', 'count'])
        sm = (agg['sum'] + gm * smoothing) / (agg['count'] + smoothing)
        enc[vai] = train_df.iloc[vai][col].map(sm).fillna(gm).values
    train_df[f'{col}_te'] = enc
    agg = train_df.groupby(col)['is_relevant'].agg(['sum', 'count'])
    sm = (agg['sum'] + gm * smoothing) / (agg['count'] + smoothing)
    test_df[f'{col}_te'] = test_df[col].map(sm).fillna(gm)
    return train_df, test_df

for col in ['promo_tool', 'tool_type', 'cost_tier', 'region', 'primary_category']:
    train, test = kfold_te(train, test, col)

inters = [('promo_tool', 'seller_tier'), ('promo_tool', 'primary_category'),
           ('promo_tool', 'region'), ('tool_type', 'seller_tier'),
           ('tool_type', 'primary_category'), ('tool_type', 'region'),
           ('cost_tier', 'seller_tier'), ('promo_tool', 'cost_tier')]
for c1, c2 in inters:
    icol = f'{c1}_X_{c2}'
    train[icol] = train[c1].astype(str) + '_' + train[c2].astype(str)
    test[icol] = test[c1].astype(str) + '_' + test[c2].astype(str)
    train, test = kfold_te(train, test, icol)

# ============================================================
# 5. Per-Query Relative Features
# ============================================================
print("Per-query relative features...")

for col in ['discount_depth', 'visibility_boost', 'inventory_synergy',
            'loyalty_synergy', 'margin_penalty', 'cross_border_fit',
            'new_seller_fit', 'seasonal_fit', 'tool_benefit', 'tool_cost_benefit',
            'seller_id_promo_tool_exp']:
    if col not in train.columns:
        continue
    train[f'{col}_qrank'] = train.groupby('query_id')[col].rank(method='dense', ascending=False)
    test[f'{col}_qrank'] = test.groupby('query_id')[col].rank(method='dense', ascending=False)
    train[f'{col}_qrel'] = train[col] - train.groupby('query_id')[col].transform('mean')
    test[f'{col}_qrel'] = test[col] - test.groupby('query_id')[col].transform('mean')

# ============================================================
# 6. Feature Selection
# ============================================================
print("Selecting features...")

drop_cols = [
    'candidate_id', 'query_id', 'snapshot_date', 'seller_id',
    'active_channels', 'recent_promo_tools', 'is_relevant',
    'region', 'primary_category', 'seller_tier', 'promo_tool',
    'tool_type', 'cost_tier',
]
for c1, c2 in inters:
    drop_cols.append(f'{c1}_X_{c2}')

feature_cols = [c for c in train.columns if c not in drop_cols]
print(f"Total features: {len(feature_cols)}")

X_train = train[feature_cols].values.astype(np.float32)
y_train = target
X_test = test[feature_cols].values.astype(np.float32)

tqids = train['query_id'].values
test_qids = test['query_id'].values
test_cids = test['candidate_id'].values

tqo = pd.Series(tqids).unique()
train_groups = train.groupby('query_id').size().reindex(tqo).values
test_qo = pd.Series(test_qids).unique()
test_groups = test.groupby('query_id').size().reindex(test_qo).values

# ============================================================
# 7. Validation
# ============================================================
print("Validation split...")

vc = pd.Timestamp('2024-05-06')
vm = train['snapshot_date'] >= vc
tm_ = ~vm

Xt, yt = X_train[tm_], y_train[tm_]
Xv, yv = X_train[vm], y_train[vm]
tq_ = tqids[tm_]; vq_ = tqids[vm]

tqo_ = pd.Series(tq_).unique()
tg_ = pd.DataFrame({'q': tq_}).groupby('q').size().reindex(tqo_).values
vqo_ = pd.Series(vq_).unique()
vg_ = pd.DataFrame({'q': vq_}).groupby('q').size().reindex(vqo_).values

# ============================================================
# 8. LightGBM LambdaRank
# ============================================================
print("\n=== LightGBM LambdaRank ===")

lgb_p = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [3],
    'boosting_type': 'gbdt', 'learning_rate': 0.02, 'num_leaves': 63,
    'max_depth': 8, 'min_child_samples': 20, 'feature_fraction': 0.75,
    'bagging_fraction': 0.85, 'bagging_freq': 5, 'lambda_l1': 0.05,
    'lambda_l2': 0.5, 'min_gain_to_split': 0.005, 'verbose': -1,
    'n_jobs': -1, 'label_gain': [0, 1],
}

td = lgb.Dataset(Xt, label=yt, group=tg_, feature_name=feature_cols)
vd = lgb.Dataset(Xv, label=yv, group=vg_, feature_name=feature_cols, reference=td)

mr = lgb.train(lgb_p, td, num_boost_round=3000, valid_sets=[vd], valid_names=['val'],
               callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])

ri = mr.best_iteration
print(f"LambdaRank best iter: {ri}")

# ============================================================
# 9. LightGBM Binary
# ============================================================
print("\n=== LightGBM Binary ===")

lgb_b = {
    'objective': 'binary', 'metric': 'binary_logloss',
    'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 8,
    'min_child_samples': 20, 'feature_fraction': 0.75,
    'bagging_fraction': 0.85, 'bagging_freq': 5, 'lambda_l1': 0.05,
    'lambda_l2': 0.5, 'verbose': -1, 'n_jobs': -1,
}

td_b = lgb.Dataset(Xt, label=yt, feature_name=feature_cols)
vd_b = lgb.Dataset(Xv, label=yv, feature_name=feature_cols, reference=td_b)

mb = lgb.train(lgb_b, td_b, num_boost_round=3000, valid_sets=[vd_b], valid_names=['val'],
               callbacks=[lgb.early_stopping(150), lgb.log_evaluation(200)])

bi = mb.best_iteration

# Compute NDCG@3
vpr = mr.predict(Xv)
vpb = mb.predict(Xv)

vdf = pd.DataFrame({'q': vq_, 'y': yv, 'sr': vpr, 'sb': vpb})
n3r = vdf.groupby('q').apply(lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['sr'].values.reshape(1,-1), k=3)).mean()
n3b = vdf.groupby('q').apply(lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['sb'].values.reshape(1,-1), k=3)).mean()

print(f"\nLambdaRank NDCG@3: {n3r:.6f}")
print(f"Binary NDCG@3:     {n3b:.6f}")

# Best blend
bw, bn = 0, 0
for w in np.arange(0, 1.01, 0.05):
    bl = w * MinMaxScaler().fit_transform(vpr.reshape(-1,1)).flatten() + (1-w) * MinMaxScaler().fit_transform(vpb.reshape(-1,1)).flatten()
    vdf['bl'] = bl
    n = vdf.groupby('q').apply(lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['bl'].values.reshape(1,-1), k=3)).mean()
    if n > bn:
        bn = n; bw = w

print(f"Best blend: w_rank={bw:.2f}, NDCG@3={bn:.6f}")

# Feature importance
imp = pd.DataFrame({'f': feature_cols, 'g': mr.feature_importance('gain')}).sort_values('g', ascending=False)
print("\nTop 20 features:")
print(imp.head(20).to_string(index=False))

# ============================================================
# 10. Final Models (Multi-Seed)
# ============================================================
print("\n=== Final Training ===")

seeds = [42, 123, 456, 789, 2024, 1337, 7777, 9999, 31415]
pr = np.zeros(len(X_test))
pb = np.zeros(len(X_test))

fd = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=feature_cols)
fb = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

for s in seeds:
    print(f"  Seed {s}...")
    p1 = lgb_p.copy(); p1['seed'] = s
    m1 = lgb.train(p1, fd, num_boost_round=int(ri * 1.1))
    pr += m1.predict(X_test)
    
    p2 = lgb_b.copy(); p2['seed'] = s
    m2 = lgb.train(p2, fb, num_boost_round=int(bi * 1.1))
    pb += m2.predict(X_test)

pr /= len(seeds); pb /= len(seeds)

prn = MinMaxScaler().fit_transform(pr.reshape(-1,1)).flatten()
pbn = MinMaxScaler().fit_transform(pb.reshape(-1,1)).flatten()
pf = bw * prn + (1 - bw) * pbn

# ============================================================
# 11. Submissions
# ============================================================
print("\nSubmissions...")

for nm, sc in [('submission.csv', pf), ('submission_rank.csv', pr), ('submission_bin.csv', pb)]:
    pd.DataFrame({'candidate_id': test_cids, 'score': sc}).to_csv(nm, index=False)
    print(f"  {nm}: score range [{sc.min():.4f}, {sc.max():.4f}]")

print(f"\n=== DONE ===")
print(f"Rank NDCG@3: {n3r:.6f}, Binary NDCG@3: {n3b:.6f}, Blend: {bn:.6f}")
print(f"Features: {len(feature_cols)}, Seeds: {len(seeds)}")
