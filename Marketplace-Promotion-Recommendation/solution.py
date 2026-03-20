"""
Marketplace Promotion Recommendation — Winning Solution
=======================================================
Learning-to-Rank with LightGBM LambdaRank + Heavy Feature Engineering
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

# Store identifiers
train_candidate_ids = train['candidate_id'].values
test_candidate_ids = test['candidate_id'].values
train_query_ids = train['query_id'].values
test_query_ids = test['query_id'].values
target = train['is_relevant'].values

# ============================================================
# 2. Feature Engineering
# ============================================================
print("Engineering features...")

def engineer_features(df, is_train=True):
    """Apply all feature engineering to a dataframe."""
    df = df.copy()
    
    # ---- 2a. Parse date features ----
    df['snapshot_date'] = pd.to_datetime(df['snapshot_date'])
    df['day_of_year'] = df['snapshot_date'].dt.dayofyear
    df['week_of_month'] = df['snapshot_date'].dt.day // 7 + 1
    
    # ---- 2b. Multi-hot encode active_channels ----
    channels = ['search', 'ads', 'social', 'affiliate', 'live']
    for ch in channels:
        df[f'channel_{ch}'] = df['active_channels'].fillna('').str.contains(ch).astype(int)
    df['n_active_channels'] = df[[f'channel_{ch}' for ch in channels]].sum(axis=1)
    
    # ---- 2c. Multi-hot encode recent_promo_tools ----
    promo_tools_list = [
        'homepage_feature', 'sponsored_search_boost', 'bundle_builder',
        'flash_sale_slot', 'loyalty_points_multiplier', 'coupon_pack',
        'free_shipping_boost', 'cashback_offer'
    ]
    for pt in promo_tools_list:
        df[f'recent_{pt}'] = df['recent_promo_tools'].fillna('').str.contains(pt).astype(int)
    df['n_recent_promo_tools'] = df[[f'recent_{pt}' for pt in promo_tools_list]].sum(axis=1)
    
    # ---- 2d. Match: is this candidate tool in seller's recent promo history? ----
    # This is basically the same as tool_recently_used but let's be explicit
    df['tool_in_recent_promos'] = df.apply(
        lambda row: 1 if pd.notna(row['recent_promo_tools']) and row['promo_tool'] in str(row['recent_promo_tools']).split('|') else 0,
        axis=1
    )
    
    # ---- 2e. Interaction features ----
    df['cross_border_x_fit'] = df['is_cross_border'] * df['cross_border_fit']
    df['new_seller_x_fit'] = (df['seller_tier'] == 'new').astype(int) * df['new_seller_fit']
    df['holiday_x_seasonal'] = df['holiday_campaign'] * df['seasonal_fit']
    df['month_end_x_discount'] = df['month_end_push'] * df['discount_depth']
    df['fulfillment_x_inventory'] = df['uses_fulfillment_service'] * df['inventory_synergy']
    df['loyalty_x_repeat'] = df['loyalty_synergy'] * df['repeat_buyer_rate']
    df['visibility_x_search'] = df['visibility_boost'] * df['search_visibility_score']
    df['discount_x_margin'] = df['discount_depth'] * df['margin_rate']
    df['margin_penalty_x_rate'] = df['margin_penalty'] * df['margin_rate']
    df['cashback_x_budget'] = df['cross_border_fit'] * df['cashback_budget_score']  # cashback budget compatibility
    
    # ---- 2f. Seller performance ratios ----
    df['orders_per_day'] = df['orders_30d'] / 30
    df['gmv_per_order'] = df['gmv_30d'] / (df['orders_30d'] + 1)
    df['ad_efficiency'] = df['gmv_30d'] / (df['ad_spend_30d'] + 1)
    df['view_to_order'] = df['orders_30d'] / (df['listing_views_30d'] + 1)
    df['revenue_per_view'] = df['gmv_30d'] / (df['listing_views_30d'] + 1)
    df['marketing_vs_fatigue'] = df['marketing_readiness_score'] - df['promotion_fatigue_30d']
    df['fill_vs_stockout'] = df['inventory_fill_rate'] - df['stockout_rate_30d']
    
    # ---- 2g. Tool attractiveness composite scores ----
    df['tool_benefit_score'] = df['visibility_boost'] + df['inventory_synergy'] + df['loyalty_synergy'] - df['margin_penalty']
    df['tool_cost_benefit'] = (df['visibility_boost'] + df['inventory_synergy']) / (df['margin_penalty'] + 0.1)
    
    # ---- 2h. Label encode categoricals ----
    cat_cols = ['region', 'primary_category', 'seller_tier', 'promo_tool', 'tool_type', 'cost_tier']
    for col in cat_cols:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    
    # ---- 2i. Seller segment indicators ----
    df['is_premium'] = (df['seller_tier'] == 'premium').astype(int)
    df['is_established'] = (df['seller_tier'] == 'established').astype(int)
    df['is_growth'] = (df['seller_tier'] == 'growth').astype(int)
    df['is_new'] = (df['seller_tier'] == 'new').astype(int)
    
    # ---- 2j. Tool type one-hot ----
    for tt in df['tool_type'].unique():
        df[f'tt_{tt}'] = (df['tool_type'] == tt).astype(int)
    
    return df


# Apply feature engineering
train_fe = engineer_features(train, is_train=True)
test_fe = engineer_features(test, is_train=False)

# ============================================================
# 3. Target Encoding (with K-fold to avoid leakage)
# ============================================================
print("Applying target encoding...")

def target_encode_column(train_df, test_df, col, target_col='is_relevant', n_splits=5, smoothing=10):
    """K-fold target encoding to avoid leakage."""
    global_mean = train_df[target_col].mean()
    
    # For train: use out-of-fold encoding
    train_df[f'{col}_target_enc'] = global_mean
    kf = GroupKFold(n_splits=n_splits)
    groups = train_df['query_id']
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(train_df, train_df[target_col], groups)):
        fold_train = train_df.iloc[train_idx]
        stats = fold_train.groupby(col)[target_col].agg(['mean', 'count'])
        smoothed = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        train_df.iloc[val_idx, train_df.columns.get_loc(f'{col}_target_enc')] = train_df.iloc[val_idx][col].map(smoothed).fillna(global_mean)
    
    # For test: use full train encoding
    stats = train_df.groupby(col)[target_col].agg(['mean', 'count'])
    smoothed = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
    test_df[f'{col}_target_enc'] = test_df[col].map(smoothed).fillna(global_mean)
    
    return train_df, test_df

# Target encode key columns
for col in ['promo_tool', 'tool_type', 'cost_tier', 'region', 'primary_category', 'seller_tier']:
    train_fe, test_fe = target_encode_column(train_fe, test_fe, col)

# ---- Target encode interactions ----
train_fe['tool_x_tier'] = train_fe['promo_tool'] + '_' + train_fe['seller_tier']
test_fe['tool_x_tier'] = test_fe['promo_tool'] + '_' + test_fe['seller_tier']
train_fe, test_fe = target_encode_column(train_fe, test_fe, 'tool_x_tier')

train_fe['tool_x_category'] = train_fe['promo_tool'] + '_' + train_fe['primary_category']
test_fe['tool_x_category'] = test_fe['promo_tool'] + '_' + test_fe['primary_category']
train_fe, test_fe = target_encode_column(train_fe, test_fe, 'tool_x_category')

train_fe['tool_x_region'] = train_fe['promo_tool'] + '_' + train_fe['region']
test_fe['tool_x_region'] = test_fe['promo_tool'] + '_' + test_fe['region']
train_fe, test_fe = target_encode_column(train_fe, test_fe, 'tool_x_region')

train_fe['type_x_tier'] = train_fe['tool_type'] + '_' + train_fe['seller_tier']
test_fe['type_x_tier'] = test_fe['tool_type'] + '_' + test_fe['seller_tier']
train_fe, test_fe = target_encode_column(train_fe, test_fe, 'type_x_tier')

train_fe['type_x_category'] = train_fe['tool_type'] + '_' + train_fe['primary_category']
test_fe['type_x_category'] = test_fe['tool_type'] + '_' + test_fe['primary_category']
train_fe, test_fe = target_encode_column(train_fe, test_fe, 'type_x_category')

# ============================================================
# 4. Per-Query Relative Features
# ============================================================
print("Computing per-query relative features...")

rank_cols = [
    'discount_depth', 'visibility_boost', 'inventory_synergy', 
    'loyalty_synergy', 'margin_penalty', 'tool_benefit_score',
    'tool_cost_benefit', 'cross_border_fit', 'new_seller_fit', 'seasonal_fit'
]

for col in rank_cols:
    # Rank within query
    train_fe[f'{col}_rank'] = train_fe.groupby('query_id')[col].rank(method='dense', ascending=False)
    test_fe[f'{col}_rank'] = test_fe.groupby('query_id')[col].rank(method='dense', ascending=False)
    
    # Relative to query mean
    train_fe[f'{col}_vs_qmean'] = train_fe[col] - train_fe.groupby('query_id')[col].transform('mean')
    test_fe[f'{col}_vs_qmean'] = test_fe[col] - test_fe.groupby('query_id')[col].transform('mean')
    
    # Relative to query max
    train_fe[f'{col}_vs_qmax'] = train_fe[col] / (train_fe.groupby('query_id')[col].transform('max') + 1e-8)
    test_fe[f'{col}_vs_qmax'] = test_fe[col] / (test_fe.groupby('query_id')[col].transform('max') + 1e-8)

# ============================================================
# 5. Seller Historical Features (Target-Based)
# ============================================================
print("Computing seller-level historical features...")

# Within training data, compute seller-level tool preference history
# For each seller, which tools have been historically relevant?
seller_tool_hist = train.groupby(['seller_id', 'promo_tool'])['is_relevant'].mean().reset_index()
seller_tool_hist.columns = ['seller_id', 'promo_tool', 'seller_tool_hist_rate']

train_fe = train_fe.merge(seller_tool_hist, on=['seller_id', 'promo_tool'], how='left')
test_fe = test_fe.merge(seller_tool_hist, on=['seller_id', 'promo_tool'], how='left')
train_fe['seller_tool_hist_rate'] = train_fe['seller_tool_hist_rate'].fillna(0.25)
test_fe['seller_tool_hist_rate'] = test_fe['seller_tool_hist_rate'].fillna(0.25)

# Seller-level overall relevance rate
seller_hist = train.groupby('seller_id')['is_relevant'].mean().reset_index()
seller_hist.columns = ['seller_id', 'seller_hist_rate']
train_fe = train_fe.merge(seller_hist, on='seller_id', how='left')
test_fe = test_fe.merge(seller_hist, on='seller_id', how='left')

# Category-tool historical rate
cat_tool_hist = train.groupby(['primary_category', 'promo_tool'])['is_relevant'].mean().reset_index()
cat_tool_hist.columns = ['primary_category', 'promo_tool', 'cat_tool_hist_rate']
train_fe = train_fe.merge(cat_tool_hist, on=['primary_category', 'promo_tool'], how='left')
test_fe = test_fe.merge(cat_tool_hist, on=['primary_category', 'promo_tool'], how='left')

# Region-tool historical rate
reg_tool_hist = train.groupby(['region', 'promo_tool'])['is_relevant'].mean().reset_index()
reg_tool_hist.columns = ['region', 'promo_tool', 'reg_tool_hist_rate']
train_fe = train_fe.merge(reg_tool_hist, on=['region', 'promo_tool'], how='left')
test_fe = test_fe.merge(reg_tool_hist, on=['region', 'promo_tool'], how='left')

# ============================================================
# 6. Select Features
# ============================================================
print("Selecting features...")

drop_cols = [
    'candidate_id', 'query_id', 'snapshot_date', 'seller_id',
    'active_channels', 'recent_promo_tools', 'is_relevant',
    'region', 'primary_category', 'seller_tier', 'promo_tool', 
    'tool_type', 'cost_tier',
    'tool_x_tier', 'tool_x_category', 'tool_x_region',
    'type_x_tier', 'type_x_category'
]

feature_cols = [c for c in train_fe.columns if c not in drop_cols]
print(f"Total features: {len(feature_cols)}")

X_train = train_fe[feature_cols].values.astype(np.float32)
y_train = target
X_test = test_fe[feature_cols].values.astype(np.float32)

# Group sizes for ranking (8 candidates per query)
train_queries = train_fe['query_id'].values
test_queries = test_fe['query_id'].values

# Get group sizes
train_query_order = pd.Series(train_queries).unique()
train_groups = train_fe.groupby('query_id').size().reindex(train_query_order).values
test_query_order = pd.Series(test_queries).unique()
test_groups = test_fe.groupby('query_id').size().reindex(test_query_order).values

# ============================================================
# 7. Time-Based Validation
# ============================================================
print("Setting up time-based validation...")

val_cutoff = '2024-05-06'
train_dates = train_fe['snapshot_date']

val_mask = train_dates >= val_cutoff
trn_mask = ~val_mask

X_trn, y_trn = X_train[trn_mask], y_train[trn_mask]
X_val, y_val = X_train[val_mask], y_train[val_mask]

trn_queries = train_queries[trn_mask]
val_queries = train_queries[val_mask]

trn_query_order = pd.Series(trn_queries).unique()
trn_groups = pd.Series(trn_queries).groupby(trn_queries).size().reindex(trn_query_order).values
val_query_order = pd.Series(val_queries).unique()
val_groups = pd.Series(val_queries).groupby(val_queries).size().reindex(val_query_order).values

# ============================================================
# 8. Train LightGBM LambdaRank
# ============================================================
print("Training LightGBM LambdaRank...")

params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'eval_at': [3],
    'boosting_type': 'gbdt',
    'learning_rate': 0.03,
    'num_leaves': 63,
    'max_depth': 7,
    'min_child_samples': 30,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.85,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 1.0,
    'min_gain_to_split': 0.01,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
    'label_gain': [0, 1],
}

# Validation run first
trn_data = lgb.Dataset(X_trn, label=y_trn, group=trn_groups, feature_name=feature_cols)
val_data = lgb.Dataset(X_val, label=y_val, group=val_groups, feature_name=feature_cols, reference=trn_data)

model_val = lgb.train(
    params,
    trn_data,
    num_boost_round=2000,
    valid_sets=[val_data],
    valid_names=['valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(100)
    ]
)

val_score = model_val.best_score['valid']['ndcg@3']
print(f"\n{'='*60}")
print(f"Validation NDCG@3: {val_score:.6f}")
print(f"Best iteration: {model_val.best_iteration}")
print(f"{'='*60}")

# ============================================================
# 9. Feature Importance
# ============================================================
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model_val.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("\nTop 30 features:")
print(importance.head(30).to_string(index=False))

# ============================================================
# 10. Train Final Model (Full Training Data, Multi-Seed)
# ============================================================
print("\nTraining final models on full data with multi-seed ensemble...")

best_iter = model_val.best_iteration
seeds = [42, 123, 456, 789, 2024]
test_preds = np.zeros(len(X_test))

for seed in seeds:
    print(f"  Training with seed {seed}...")
    params_full = params.copy()
    params_full['seed'] = seed
    
    full_data = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=feature_cols)
    
    model_full = lgb.train(
        params_full,
        full_data,
        num_boost_round=int(best_iter * 1.1),  # Slightly more iterations for full data
    )
    
    preds = model_full.predict(X_test)
    test_preds += preds

test_preds /= len(seeds)

# ============================================================
# 11. Generate Submission
# ============================================================
print("Generating submission...")

submission = pd.DataFrame({
    'candidate_id': test_candidate_ids,
    'score': test_preds
})

submission.to_csv('submission.csv', index=False)
print(f"Submission saved: {submission.shape[0]} rows")
print(f"Score range: [{submission['score'].min():.4f}, {submission['score'].max():.4f}]")
print("\nDone!")
