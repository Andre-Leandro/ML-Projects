"""
V4 — Final Ensemble: LightGBM (LambdaRank + Binary) + CatBoost
Based on V2 (best performing), adds CatBoost and wider multi-seed ensemble.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, Pool
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load and Engineer Features (same as V2)
# ============================================================
print("Loading data...")
train = pd.read_csv('dataset/public/train.csv')
test = pd.read_csv('dataset/public/test.csv')

target = train['is_relevant'].values
train['snapshot_date'] = pd.to_datetime(train['snapshot_date'])
test['snapshot_date'] = pd.to_datetime(test['snapshot_date'])

print("Engineering features...")

def add_features(df):
    df = df.copy()
    df['day_of_year'] = df['snapshot_date'].dt.dayofyear
    df['week_of_month'] = df['snapshot_date'].dt.day // 7 + 1
    
    for ch in ['search', 'ads', 'social', 'affiliate', 'live']:
        df[f'ch_{ch}'] = df['active_channels'].fillna('').str.contains(ch).astype(int)
    df['n_channels'] = df[['ch_search','ch_ads','ch_social','ch_affiliate','ch_live']].sum(axis=1)
    
    tools = ['homepage_feature', 'sponsored_search_boost', 'bundle_builder',
             'flash_sale_slot', 'loyalty_points_multiplier', 'coupon_pack',
             'free_shipping_boost', 'cashback_offer']
    for pt in tools:
        df[f'rpt_{pt}'] = df['recent_promo_tools'].fillna('').str.contains(pt).astype(int)
    df['n_recent_tools'] = df[[f'rpt_{pt}' for pt in tools]].sum(axis=1)
    
    def check_match(row):
        if pd.isna(row['recent_promo_tools']): return 0
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
    
    df['gmv_per_order'] = df['gmv_30d'] / (df['orders_30d'] + 1)
    df['ad_efficiency'] = df['gmv_30d'] / (df['ad_spend_30d'] + 1)
    df['view_to_order'] = df['orders_30d'] / (df['listing_views_30d'] + 1)
    df['readiness_minus_fatigue'] = df['marketing_readiness_score'] - df['promotion_fatigue_30d']
    df['fill_minus_stockout'] = df['inventory_fill_rate'] - df['stockout_rate_30d']
    df['ad_per_order'] = df['ad_spend_30d'] / (df['orders_30d'] + 1)
    df['margin_gmv'] = df['margin_rate'] * df['gmv_30d']
    
    df['tool_benefit'] = df['visibility_boost'] + df['inventory_synergy'] + df['loyalty_synergy'] - df['margin_penalty']
    df['tool_cost_benefit'] = (df['visibility_boost'] + df['inventory_synergy']) / (df['margin_penalty'] + 0.1)
    
    for col in ['region', 'primary_category', 'seller_tier', 'promo_tool', 'tool_type', 'cost_tier']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    
    for t in ['premium', 'established', 'growth', 'new']:
        df[f'is_{t}'] = (df['seller_tier'] == t).astype(int)
    for tt in ['visibility', 'bundle', 'event', 'retention', 'shipping', 'discount']:
        df[f'tt_{tt}'] = (df['tool_type'] == tt).astype(int)
    
    return df

train = add_features(train)
test = add_features(test)

# ============================================================
# 2. Target Encoding
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
# 3. Seller Historical Aggregates
# ============================================================
print("Seller aggregates...")

st_agg = train.groupby(['seller_id', 'promo_tool'])['is_relevant'].agg(['mean', 'count']).reset_index()
st_agg.columns = ['seller_id', 'promo_tool', 'seller_tool_rate', 'seller_tool_count']
train = train.merge(st_agg, on=['seller_id', 'promo_tool'], how='left')
test = test.merge(st_agg, on=['seller_id', 'promo_tool'], how='left')
test['seller_tool_rate'] = test['seller_tool_rate'].fillna(0.25)
test['seller_tool_count'] = test['seller_tool_count'].fillna(0)

ct_agg = train.groupby(['primary_category', 'promo_tool'])['is_relevant'].mean().reset_index()
ct_agg.columns = ['primary_category', 'promo_tool', 'cat_tool_rate']
train = train.merge(ct_agg, on=['primary_category', 'promo_tool'], how='left')
test = test.merge(ct_agg, on=['primary_category', 'promo_tool'], how='left')

rt_agg = train.groupby(['region', 'promo_tool'])['is_relevant'].mean().reset_index()
rt_agg.columns = ['region', 'promo_tool', 'reg_tool_rate']
train = train.merge(rt_agg, on=['region', 'promo_tool'], how='left')
test = test.merge(rt_agg, on=['region', 'promo_tool'], how='left')

s_agg = train.groupby('seller_id')['is_relevant'].mean().reset_index()
s_agg.columns = ['seller_id', 'seller_rate']
train = train.merge(s_agg, on='seller_id', how='left')
test = test.merge(s_agg, on='seller_id', how='left')

# ============================================================
# 4. Per-Query Features
# ============================================================
print("Per-query features...")

for col in ['discount_depth', 'visibility_boost', 'inventory_synergy',
            'loyalty_synergy', 'margin_penalty', 'cross_border_fit',
            'new_seller_fit', 'seasonal_fit', 'tool_benefit', 'tool_cost_benefit',
            'seller_tool_rate']:
    train[f'{col}_qrank'] = train.groupby('query_id')[col].rank(method='dense', ascending=False)
    test[f'{col}_qrank'] = test.groupby('query_id')[col].rank(method='dense', ascending=False)
    train[f'{col}_qrel'] = train[col] - train.groupby('query_id')[col].transform('mean')
    test[f'{col}_qrel'] = test[col] - test.groupby('query_id')[col].transform('mean')

# ============================================================
# 5. Feature Selection
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
print(f"Features: {len(feature_cols)}")

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

# Validation split
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
# 6. LightGBM LambdaRank - param sweep
# ============================================================
print("\n=== LightGBM LambdaRank (param sweep) ===")

base_lgb = {
    'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [3],
    'boosting_type': 'gbdt', 'verbose': -1, 'n_jobs': -1, 'label_gain': [0, 1],
    'feature_pre_filter': False,
}

param_configs = [
    {'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 8, 'min_child_samples': 20,
     'feature_fraction': 0.75, 'bagging_fraction': 0.85, 'bagging_freq': 5,
     'lambda_l1': 0.05, 'lambda_l2': 0.5, 'min_gain_to_split': 0.005},
    {'learning_rate': 0.015, 'num_leaves': 127, 'max_depth': 10, 'min_child_samples': 30,
     'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 3,
     'lambda_l1': 0.1, 'lambda_l2': 1.0, 'min_gain_to_split': 0.01},
    {'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6, 'min_child_samples': 15,
     'feature_fraction': 0.8, 'bagging_fraction': 0.9, 'bagging_freq': 5,
     'lambda_l1': 0.02, 'lambda_l2': 0.2, 'min_gain_to_split': 0.002},
]

best_rank_models = []
td = lgb.Dataset(Xt, label=yt, group=tg_, feature_name=feature_cols)
vd = lgb.Dataset(Xv, label=yv, group=vg_, feature_name=feature_cols, reference=td)

for i, pc in enumerate(param_configs):
    p = {**base_lgb, **pc}
    m = lgb.train(p, td, num_boost_round=3000, valid_sets=[vd], valid_names=['val'],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
    score = m.best_score['val']['ndcg@3']
    best_rank_models.append((m, p, score, m.best_iteration))
    print(f"Config {i}: NDCG@3={score:.6f}, iter={m.best_iteration}")

# ============================================================
# 7. LightGBM Binary - param sweep
# ============================================================
print("\n=== LightGBM Binary (param sweep) ===")

base_bin = {
    'objective': 'binary', 'metric': 'binary_logloss', 'verbose': -1, 'n_jobs': -1,
    'feature_pre_filter': False,
}

best_bin_models = []
td_b = lgb.Dataset(Xt, label=yt, feature_name=feature_cols)
vd_b = lgb.Dataset(Xv, label=yv, feature_name=feature_cols, reference=td_b)

for i, pc in enumerate(param_configs):
    pc2 = {k: v for k, v in pc.items() if k != 'min_gain_to_split'}
    p = {**base_bin, **pc2}
    m = lgb.train(p, td_b, num_boost_round=3000, valid_sets=[vd_b], valid_names=['val'],
                  callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
    
    vp = m.predict(Xv)
    vdf = pd.DataFrame({'q': vq_, 'y': yv, 's': vp})
    n3 = vdf.groupby('q').apply(lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['s'].values.reshape(1,-1), k=3)).mean()
    best_bin_models.append((m, p, n3, m.best_iteration))
    print(f"Config {i}: NDCG@3={n3:.6f}, iter={m.best_iteration}")

# ============================================================
# 8. CatBoost Classifier
# ============================================================
print("\n=== CatBoost ===")

cb = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3,
    eval_metric='Logloss',
    random_seed=42,
    verbose=200,
    early_stopping_rounds=100,
)

cb.fit(Xt, yt, eval_set=(Xv, yv), verbose=200)
cb_preds_val = cb.predict_proba(Xv)[:, 1]

vdf_cb = pd.DataFrame({'q': vq_, 'y': yv, 's': cb_preds_val})
cb_n3 = vdf_cb.groupby('q').apply(lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['s'].values.reshape(1,-1), k=3)).mean()
cb_iter = cb.best_iteration_
print(f"CatBoost NDCG@3: {cb_n3:.6f}, iter={cb_iter}")

# ============================================================
# 9. Optimal Ensemble Weights 
# ============================================================
print("\n=== Finding optimal ensemble weights ===")

# Collect all validation predictions
all_val_preds = []
all_labels = []

for m, p, s, it in best_rank_models:
    vp = m.predict(Xv)
    all_val_preds.append(MinMaxScaler().fit_transform(vp.reshape(-1,1)).flatten())
    all_labels.append(f'rank_{s:.4f}')

for m, p, s, it in best_bin_models:
    vp = m.predict(Xv)
    all_val_preds.append(MinMaxScaler().fit_transform(vp.reshape(-1,1)).flatten())
    all_labels.append(f'bin_{s:.4f}')

all_val_preds.append(MinMaxScaler().fit_transform(cb_preds_val.reshape(-1,1)).flatten())
all_labels.append(f'cb_{cb_n3:.4f}')

# Simple average of all models
avg_pred = np.mean(all_val_preds, axis=0)
vdf_avg = pd.DataFrame({'q': vq_, 'y': yv, 's': avg_pred})
avg_n3 = vdf_avg.groupby('q').apply(lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['s'].values.reshape(1,-1), k=3)).mean()
print(f"Simple average all models NDCG@3: {avg_n3:.6f}")

# Try weighted: rank models at 40%, bin models at 40%, catboost at 20%
n_rank = len(best_rank_models)
n_bin = len(best_bin_models)

w_preds = np.zeros(len(Xv))
for i in range(n_rank):
    w_preds += 0.4 / n_rank * all_val_preds[i]
for i in range(n_bin):
    w_preds += 0.4 / n_bin * all_val_preds[n_rank + i]
w_preds += 0.2 * all_val_preds[-1]

vdf_w = pd.DataFrame({'q': vq_, 'y': yv, 's': w_preds})
w_n3 = vdf_w.groupby('q').apply(lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['s'].values.reshape(1,-1), k=3)).mean()
print(f"Weighted (40/40/20) NDCG@3: {w_n3:.6f}")

# Just pick the best scheme
best_ensemble_score = max(avg_n3, w_n3)
use_weighted = w_n3 > avg_n3
print(f"Best ensemble: {'weighted' if use_weighted else 'avg'} with NDCG@3={best_ensemble_score:.6f}")

# ============================================================
# 10. Final Training (Multi-Seed)
# ============================================================
print("\n=== Final Training ===")

seeds = [42, 123, 456, 789, 2024, 1337, 7777, 9999, 31415]
all_test_preds = {label: np.zeros(len(X_test)) for label in all_labels}

fd = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=feature_cols)
fb = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

for s in seeds:
    print(f"  Seed {s}...")
    
    # Rank models
    for i, (m_val, p, sc, it) in enumerate(best_rank_models):
        p_f = p.copy(); p_f['seed'] = s
        m = lgb.train(p_f, fd, num_boost_round=int(it * 1.1))
        all_test_preds[all_labels[i]] += m.predict(X_test)
    
    # Binary models
    for i, (m_val, p, sc, it) in enumerate(best_bin_models):
        p_f = p.copy(); p_f['seed'] = s
        m = lgb.train(p_f, fb, num_boost_round=int(it * 1.1))
        all_test_preds[all_labels[n_rank + i]] += m.predict(X_test)
    
    # CatBoost
    cb_f = CatBoostClassifier(
        iterations=int(cb_iter * 1.1), learning_rate=0.05, depth=8,
        l2_leaf_reg=3, random_seed=s, verbose=0
    )
    cb_f.fit(X_train, y_train)
    all_test_preds[all_labels[-1]] += cb_f.predict_proba(X_test)[:, 1]

for label in all_labels:
    all_test_preds[label] /= len(seeds)

# Normalize all predictions
for label in all_labels:
    all_test_preds[label] = MinMaxScaler().fit_transform(all_test_preds[label].reshape(-1,1)).flatten()

# Generate final predictions
if use_weighted:
    final_pred = np.zeros(len(X_test))
    for i in range(n_rank):
        final_pred += 0.4 / n_rank * all_test_preds[all_labels[i]]
    for i in range(n_bin):
        final_pred += 0.4 / n_bin * all_test_preds[all_labels[n_rank + i]]
    final_pred += 0.2 * all_test_preds[all_labels[-1]]
else:
    final_pred = np.mean([all_test_preds[l] for l in all_labels], axis=0)

# Also generate individual submissions
print("\n=== Submissions ===")
pd.DataFrame({'candidate_id': test_cids, 'score': final_pred}).to_csv('working/submission.csv', index=False)
print(f"submission.csv (best ensemble): [{final_pred.min():.4f}, {final_pred.max():.4f}]")

# Simple average as backup
avg_final = np.mean([all_test_preds[l] for l in all_labels], axis=0)
pd.DataFrame({'candidate_id': test_cids, 'score': avg_final}).to_csv('working/submission_avg.csv', index=False)
print(f"submission_avg.csv (simple avg): [{avg_final.min():.4f}, {avg_final.max():.4f}]")

# Best single rank model
best_rank_idx = max(range(n_rank), key=lambda i: best_rank_models[i][2])
pd.DataFrame({'candidate_id': test_cids, 'score': all_test_preds[all_labels[best_rank_idx]]}).to_csv('working/submission_rank.csv', index=False)
print(f"submission_rank.csv (best rank)")

# Best single binary model
best_bin_idx = max(range(n_bin), key=lambda i: best_bin_models[i][2])
pd.DataFrame({'candidate_id': test_cids, 'score': all_test_preds[all_labels[n_rank + best_bin_idx]]}).to_csv('working/submission_bin.csv', index=False)
print(f"submission_bin.csv (best binary)")

print(f"\n=== SUMMARY ===")
for label in all_labels:
    print(f"  {label}")
print(f"Ensemble NDCG@3: {best_ensemble_score:.6f}")
print(f"Seeds: {len(seeds)}")
print(f"Features: {len(feature_cols)}")
print("DONE!")
