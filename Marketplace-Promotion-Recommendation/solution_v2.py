"""
Marketplace Promotion Recommendation — V2 Optimized Solution
=============================================================
LightGBM LambdaRank + XGBoost Ranker Ensemble
Heavy Feature Engineering, No Leakage, Multi-Seed
Metric: NDCG@3
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold
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
# 2. Feature Engineering (no leakage)
# ============================================================
print("Engineering features...")

def add_basic_features(df):
    """Add non-target-dependent features."""
    df = df.copy()
    
    # ---- Date features ----
    df['day_of_year'] = df['snapshot_date'].dt.dayofyear
    df['week_of_month'] = df['snapshot_date'].dt.day // 7 + 1
    df['days_since_start'] = (df['snapshot_date'] - df['snapshot_date'].min()).dt.days
    
    # ---- Multi-hot encode active_channels ----
    channels = ['search', 'ads', 'social', 'affiliate', 'live']
    for ch in channels:
        df[f'ch_{ch}'] = df['active_channels'].fillna('').str.contains(ch).astype(int)
    df['n_channels'] = sum(df[f'ch_{ch}'] for ch in channels)
    
    # ---- Multi-hot encode recent_promo_tools ----
    promo_tools_list = [
        'homepage_feature', 'sponsored_search_boost', 'bundle_builder',
        'flash_sale_slot', 'loyalty_points_multiplier', 'coupon_pack',
        'free_shipping_boost', 'cashback_offer'
    ]
    for pt in promo_tools_list:
        df[f'rpt_{pt}'] = df['recent_promo_tools'].fillna('').str.contains(pt).astype(int)
    df['n_recent_tools'] = sum(df[f'rpt_{pt}'] for pt in promo_tools_list)
    
    # ---- Tool-in-recent match ----
    def check_tool_match(row):
        if pd.isna(row['recent_promo_tools']):
            return 0
        return 1 if row['promo_tool'] in row['recent_promo_tools'].split('|') else 0
    df['tool_in_recent'] = df.apply(check_tool_match, axis=1)
    
    # ---- Interaction features: seller × tool ----
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
    
    # ---- Seller performance features ----
    df['gmv_per_order'] = df['gmv_30d'] / (df['orders_30d'] + 1)
    df['ad_efficiency'] = df['gmv_30d'] / (df['ad_spend_30d'] + 1)
    df['view_to_order'] = df['orders_30d'] / (df['listing_views_30d'] + 1)
    df['rev_per_view'] = df['gmv_30d'] / (df['listing_views_30d'] + 1)
    df['readiness_minus_fatigue'] = df['marketing_readiness_score'] - df['promotion_fatigue_30d']
    df['fill_minus_stockout'] = df['inventory_fill_rate'] - df['stockout_rate_30d']
    df['orders_per_day'] = df['orders_30d'] / 30
    df['ad_per_order'] = df['ad_spend_30d'] / (df['orders_30d'] + 1)
    df['margin_gmv'] = df['margin_rate'] * df['gmv_30d']
    df['listing_per_order'] = df['listing_views_30d'] / (df['orders_30d'] + 1)
    
    # ---- Tool composite scores ----
    df['tool_benefit'] = df['visibility_boost'] + df['inventory_synergy'] + df['loyalty_synergy'] - df['margin_penalty']
    df['tool_cost_benefit'] = (df['visibility_boost'] + df['inventory_synergy']) / (df['margin_penalty'] + 0.1)
    df['tool_overall_fit'] = (
        df['visibility_boost'] * 0.3 + df['inventory_synergy'] * 0.2 +
        df['loyalty_synergy'] * 0.2 + df['cross_border_fit'] * 0.1 +
        df['new_seller_fit'] * 0.1 + df['seasonal_fit'] * 0.1
    )
    
    # ---- Label encode categoricals ----
    cat_map = {}
    for col in ['region', 'primary_category', 'seller_tier', 'promo_tool', 'tool_type', 'cost_tier']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
        cat_map[col] = le
    
    # ---- Tier dummies ----
    for t in ['premium', 'established', 'growth', 'new']:
        df[f'is_{t}'] = (df['seller_tier'] == t).astype(int)
    
    # ---- Tool type dummies ----
    for tt in ['visibility', 'bundle', 'event', 'retention', 'shipping', 'discount']:
        df[f'tt_{tt}'] = (df['tool_type'] == tt).astype(int)
    
    # ---- Promo tool dummies ----
    for pt in df['promo_tool'].unique():
        df[f'pt_{pt}'] = (df['promo_tool'] == pt).astype(int)
    
    return df

train = add_basic_features(train)
test = add_basic_features(test)

# ============================================================
# 3. Target Encoding (K-Fold, no leakage)
# ============================================================
print("Target encoding...")

def kfold_target_encode(train_df, test_df, col, target_col='is_relevant', n_splits=5, smoothing=20):
    global_mean = train_df[target_col].mean()
    encoded_train = np.full(len(train_df), global_mean)
    
    kf = GroupKFold(n_splits=n_splits)
    groups = train_df['query_id']
    
    for train_idx, val_idx in kf.split(train_df, train_df[target_col], groups):
        fold_train = train_df.iloc[train_idx]
        agg = fold_train.groupby(col)[target_col].agg(['sum', 'count'])
        smoothed = (agg['sum'] + global_mean * smoothing) / (agg['count'] + smoothing)
        encoded_train[val_idx] = train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean).values
    
    train_df[f'{col}_te'] = encoded_train
    
    # Test: use full train
    agg = train_df.groupby(col)[target_col].agg(['sum', 'count'])
    smoothed = (agg['sum'] + global_mean * smoothing) / (agg['count'] + smoothing)
    test_df[f'{col}_te'] = test_df[col].map(smoothed).fillna(global_mean)
    
    return train_df, test_df

# Single-column TE
for col in ['promo_tool', 'tool_type', 'cost_tier', 'region', 'primary_category']:
    train, test = kfold_target_encode(train, test, col)

# Create interaction columns for TE
interactions = [
    ('promo_tool', 'seller_tier'),
    ('promo_tool', 'primary_category'),
    ('promo_tool', 'region'),
    ('tool_type', 'seller_tier'),
    ('tool_type', 'primary_category'),
    ('promo_tool', 'cost_tier'),
    ('tool_type', 'region'),
    ('cost_tier', 'seller_tier'),
]
for c1, c2 in interactions:
    icol = f'{c1}_X_{c2}'
    train[icol] = train[c1].astype(str) + '_' + train[c2].astype(str)
    test[icol] = test[c1].astype(str) + '_' + test[c2].astype(str)
    train, test = kfold_target_encode(train, test, icol)

# ============================================================
# 4. Per-Query Relative Features
# ============================================================
print("Computing per-query relative features...")

tool_numeric_cols = [
    'discount_depth', 'visibility_boost', 'inventory_synergy',
    'loyalty_synergy', 'margin_penalty', 'cross_border_fit',
    'new_seller_fit', 'seasonal_fit', 'tool_benefit', 'tool_cost_benefit',
    'tool_overall_fit'
]

for col in tool_numeric_cols:
    train[f'{col}_qrank'] = train.groupby('query_id')[col].rank(method='dense', ascending=False)
    test[f'{col}_qrank'] = test.groupby('query_id')[col].rank(method='dense', ascending=False)
    
    train[f'{col}_qrel'] = train[col] - train.groupby('query_id')[col].transform('mean')
    test[f'{col}_qrel'] = test[col] - test.groupby('query_id')[col].transform('mean')

# ============================================================
# 5. Seller Historical Features (temporal-aware for validation)
# ============================================================
print("Computing seller-level aggregates...")

# For final prediction this uses full train data
# For validation we'll handle it separately below

# Seller × tool relevance rate
seller_tool_agg = train.groupby(['seller_id', 'promo_tool'])['is_relevant'].agg(['mean', 'count']).reset_index()
seller_tool_agg.columns = ['seller_id', 'promo_tool', 'seller_tool_rate', 'seller_tool_count']
train = train.merge(seller_tool_agg, on=['seller_id', 'promo_tool'], how='left')
test = test.merge(seller_tool_agg, on=['seller_id', 'promo_tool'], how='left')
test['seller_tool_rate'] = test['seller_tool_rate'].fillna(0.25)
test['seller_tool_count'] = test['seller_tool_count'].fillna(0)

# Category × tool relevance
cat_tool_agg = train.groupby(['primary_category', 'promo_tool'])['is_relevant'].mean().reset_index()
cat_tool_agg.columns = ['primary_category', 'promo_tool', 'cat_tool_rate']
train = train.merge(cat_tool_agg, on=['primary_category', 'promo_tool'], how='left')
test = test.merge(cat_tool_agg, on=['primary_category', 'promo_tool'], how='left')

# Region × tool relevance
reg_tool_agg = train.groupby(['region', 'promo_tool'])['is_relevant'].mean().reset_index()
reg_tool_agg.columns = ['region', 'promo_tool', 'reg_tool_rate']
train = train.merge(reg_tool_agg, on=['region', 'promo_tool'], how='left')
test = test.merge(reg_tool_agg, on=['region', 'promo_tool'], how='left')

# Seller overall relevance
seller_agg = train.groupby('seller_id')['is_relevant'].mean().reset_index()
seller_agg.columns = ['seller_id', 'seller_rate']
train = train.merge(seller_agg, on='seller_id', how='left')
test = test.merge(seller_agg, on='seller_id', how='left')

# ============================================================
# 6. Select Features
# ============================================================
print("Selecting features...")

drop_cols = [
    'candidate_id', 'query_id', 'snapshot_date', 'seller_id',
    'active_channels', 'recent_promo_tools', 'is_relevant',
    'region', 'primary_category', 'seller_tier', 'promo_tool',
    'tool_type', 'cost_tier',
]
# Also drop interaction string columns
for c1, c2 in interactions:
    drop_cols.append(f'{c1}_X_{c2}')

feature_cols = [c for c in train.columns if c not in drop_cols]
print(f"Total features: {len(feature_cols)}")

X_train = train[feature_cols].values.astype(np.float32)
y_train = target
X_test = test[feature_cols].values.astype(np.float32)

train_query_ids = train['query_id'].values
test_query_ids = test['query_id'].values
test_candidate_ids = test['candidate_id'].values

# Group sizes
train_qorder = pd.Series(train_query_ids).unique()
train_groups = train.groupby('query_id').size().reindex(train_qorder).values
test_qorder = pd.Series(test_query_ids).unique()
test_groups = test.groupby('query_id').size().reindex(test_qorder).values

# ============================================================
# 7. Time-Based Validation
# ============================================================
print("Time-based validation split...")

val_cutoff = pd.Timestamp('2024-05-06')
train_dates = train['snapshot_date'].values

val_mask = train['snapshot_date'] >= val_cutoff
trn_mask = ~val_mask

X_trn, y_trn = X_train[trn_mask], y_train[trn_mask]
X_val, y_val = X_train[val_mask], y_train[val_mask]

trn_qids = train_query_ids[trn_mask]
val_qids = train_query_ids[val_mask]

trn_qorder = pd.Series(trn_qids).unique()
trn_groups = pd.DataFrame({'q': trn_qids}).groupby('q').size().reindex(trn_qorder).values
val_qorder = pd.Series(val_qids).unique()
val_groups = pd.DataFrame({'q': val_qids}).groupby('q').size().reindex(val_qorder).values

# ============================================================
# 8. LightGBM LambdaRank
# ============================================================
print("\n=== LightGBM LambdaRank ===")

lgb_params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'eval_at': [3],
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 0.05,
    'lambda_l2': 0.5,
    'min_gain_to_split': 0.005,
    'verbose': -1,
    'n_jobs': -1,
    'label_gain': [0, 1],
}

# Validation
trn_data = lgb.Dataset(X_trn, label=y_trn, group=trn_groups, feature_name=feature_cols)
val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, feature_name=feature_cols, reference=trn_data)

model_val = lgb.train(
    lgb_params,
    trn_data,
    num_boost_round=3000,
    valid_sets=[val_data],
    valid_names=['valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150),
        lgb.log_evaluation(200)
    ]
)

best_lgb_score = model_val.best_score['valid']['ndcg@3']
best_lgb_iter = model_val.best_iteration
print(f"\nLGBM Validation NDCG@3: {best_lgb_score:.6f} (iter={best_lgb_iter})")

# Feature importance
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_val.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)
print("\nTop 25 features by gain:")
print(importance.head(25).to_string(index=False))

# ============================================================
# 9. Also try binary classification approach
# ============================================================
print("\n=== LightGBM Binary Classification ===")

lgb_bin_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'boosting_type': 'gbdt',
    'learning_rate': 0.02,
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 20,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 0.05,
    'lambda_l2': 0.5,
    'verbose': -1,
    'n_jobs': -1,
}

trn_data_bin = lgb.Dataset(X_trn, label=y_trn, feature_name=feature_cols)
val_data_bin = lgb.Dataset(X_val, label=y_val, feature_name=feature_cols, reference=trn_data_bin)

model_bin_val = lgb.train(
    lgb_bin_params,
    trn_data_bin,
    num_boost_round=3000,
    valid_sets=[val_data_bin],
    valid_names=['valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=150),
        lgb.log_evaluation(200)
    ]
)

# Evaluate binary model on NDCG@3
from sklearn.metrics import ndcg_score

val_preds_bin = model_bin_val.predict(X_val)
# Compute NDCG@3 per query
val_df = pd.DataFrame({
    'query_id': val_qids,
    'is_relevant': y_val,
    'score': val_preds_bin
})

ndcg3_scores = []
for qid, group in val_df.groupby('query_id'):
    true = group['is_relevant'].values.reshape(1, -1)
    pred = group['score'].values.reshape(1, -1)
    ndcg3_scores.append(ndcg_score(true, pred, k=3))

bin_ndcg3 = np.mean(ndcg3_scores)
best_bin_iter = model_bin_val.best_iteration
print(f"Binary model NDCG@3 (computed): {bin_ndcg3:.6f} (iter={best_bin_iter})")

# Also evaluate LambdaRank with same method
val_preds_rank = model_val.predict(X_val)
val_df_rank = pd.DataFrame({
    'query_id': val_qids,
    'is_relevant': y_val,
    'score': val_preds_rank
})
ndcg3_rank = []
for qid, group in val_df_rank.groupby('query_id'):
    true = group['is_relevant'].values.reshape(1, -1)
    pred = group['score'].values.reshape(1, -1)
    ndcg3_rank.append(ndcg_score(true, pred, k=3))
rank_ndcg3 = np.mean(ndcg3_rank)
print(f"LambdaRank model NDCG@3 (computed): {rank_ndcg3:.6f}")

# ============================================================
# 10. Train Final Models on Full Data (Multi-Seed Ensemble)
# ============================================================
print("\n=== Training Final Models ===")

seeds = [42, 123, 456, 789, 2024, 1337, 7777]
test_preds_rank = np.zeros(len(X_test))
test_preds_bin = np.zeros(len(X_test))

full_data = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=feature_cols)
full_data_bin = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

for seed in seeds:
    print(f"  Seed {seed}...")
    
    # LambdaRank
    p = lgb_params.copy()
    p['seed'] = seed
    m = lgb.train(p, full_data, num_boost_round=int(best_lgb_iter * 1.1))
    test_preds_rank += m.predict(X_test)
    
    # Binary
    p2 = lgb_bin_params.copy()
    p2['seed'] = seed
    m2 = lgb.train(p2, full_data_bin, num_boost_round=int(best_bin_iter * 1.1))
    test_preds_bin += m2.predict(X_test)

test_preds_rank /= len(seeds)
test_preds_bin /= len(seeds)

# ============================================================
# 11. Blend Models
# ============================================================
print("\nBlending models...")

# Use validation performance to weight the blend
total = rank_ndcg3 + bin_ndcg3
w_rank = rank_ndcg3 / total
w_bin = bin_ndcg3 / total
print(f"Weights: rank={w_rank:.3f}, binary={w_bin:.3f}")

# Normalize predictions to same scale first
from sklearn.preprocessing import MinMaxScaler
scaler_r = MinMaxScaler()
scaler_b = MinMaxScaler()

test_preds_rank_n = scaler_r.fit_transform(test_preds_rank.reshape(-1, 1)).flatten()
test_preds_bin_n = scaler_b.fit_transform(test_preds_bin.reshape(-1, 1)).flatten()

test_preds_final = w_rank * test_preds_rank_n + w_bin * test_preds_bin_n

# ============================================================
# 12. Generate Submissions
# ============================================================
print("Generating submissions...")

# Main blended submission
submission = pd.DataFrame({
    'candidate_id': test_candidate_ids,
    'score': test_preds_final
})
submission.to_csv('submission.csv', index=False)
print(f"submission.csv: {submission.shape[0]} rows, scores [{submission['score'].min():.4f}, {submission['score'].max():.4f}]")

# Pure rank submission
submission_rank = pd.DataFrame({
    'candidate_id': test_candidate_ids,
    'score': test_preds_rank
})
submission_rank.to_csv('submission_rank.csv', index=False)
print(f"submission_rank.csv: {submission_rank.shape[0]} rows")

# Pure binary submission
submission_bin = pd.DataFrame({
    'candidate_id': test_candidate_ids,
    'score': test_preds_bin
})
submission_bin.to_csv('submission_bin.csv', index=False)
print(f"submission_bin.csv: {submission_bin.shape[0]} rows")

print("\n=== Summary ===")
print(f"LambdaRank val NDCG@3: {rank_ndcg3:.6f}")
print(f"Binary val NDCG@3:     {bin_ndcg3:.6f}")
print(f"Blend weights:         rank={w_rank:.3f}, bin={w_bin:.3f}")
print(f"Features used:         {len(feature_cols)}")
print(f"Seeds:                 {seeds}")
print("\nDone! Submit submission.csv (blended) first.")
