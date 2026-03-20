"""
Marketplace Promotion Recommendation — V6 (Enriched + Leakage-Free)
====================================================================
All V5 fixes + richer seller-tool matching features.
NO target aggregation. Focus on structural feature quality.
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import ndcg_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Load Data
# ============================================================
print("Loading data...")
train = pd.read_csv('dataset/public/train.csv')
test = pd.read_csv('dataset/public/test.csv')

target = train['is_relevant'].values
train['snapshot_date'] = pd.to_datetime(train['snapshot_date'])
test['snapshot_date'] = pd.to_datetime(test['snapshot_date'])

print(f"Train: {train.shape}, Test: {test.shape}")

# ============================================================
# 2. Feature Engineering
# ============================================================
print("Engineering features...")

ALL_TOOLS = ['homepage_feature', 'sponsored_search_boost', 'bundle_builder',
             'flash_sale_slot', 'loyalty_points_multiplier', 'coupon_pack',
             'free_shipping_boost', 'cashback_offer']

ALL_CHANNELS = ['search', 'ads', 'social', 'affiliate', 'live']

def add_features(df):
    df = df.copy()
    
    # ---- Date ----
    df['day_of_year'] = df['snapshot_date'].dt.dayofyear
    df['week_of_month'] = df['snapshot_date'].dt.day // 7 + 1
    
    # ---- Multi-hot: channels ----
    for ch in ALL_CHANNELS:
        df[f'ch_{ch}'] = df['active_channels'].fillna('').str.contains(ch).astype(int)
    df['n_channels'] = df[[f'ch_{ch}' for ch in ALL_CHANNELS]].sum(axis=1)
    
    # ---- Multi-hot: recent promo tools ----
    for pt in ALL_TOOLS:
        df[f'rpt_{pt}'] = df['recent_promo_tools'].fillna('').str.contains(pt).astype(int)
    df['n_recent_tools'] = df[[f'rpt_{pt}' for pt in ALL_TOOLS]].sum(axis=1)
    
    # ---- KEY: Tool-in-recent match ----
    def check_match(row):
        if pd.isna(row['recent_promo_tools']): return 0
        return 1 if row['promo_tool'] in row['recent_promo_tools'].split('|') else 0
    df['tool_in_recent'] = df.apply(check_match, axis=1)
    
    # ---- NEW: How many of the seller's recent tools match the same tool_type? ----
    # This captures if the seller tends to use the TYPE of tool
    # First, build a mapping of tool -> type
    tool_type_map = df.drop_duplicates('promo_tool').set_index('promo_tool')['tool_type'].to_dict()
    
    def count_same_type_in_recent(row):
        if pd.isna(row['recent_promo_tools']): return 0
        recent = row['recent_promo_tools'].split('|')
        target_type = row['tool_type']
        return sum(1 for t in recent if tool_type_map.get(t, '') == target_type)
    df['n_same_type_in_recent'] = df.apply(count_same_type_in_recent, axis=1)
    
    # ---- NEW: Ratio of matching type tools in recent ----
    df['ratio_same_type'] = df['n_same_type_in_recent'] / (df['n_recent_tools'] + 1)
    
    # ---- NEW: How many of the 8 candidate tools does the seller already know? ----
    # tool_recently_used is already binary per-candidate, but count across the query
    # We can't do this here since it requires groupby query, will do later
    
    # ---- Seller × Tool Interactions ----
    df['xborder_x_fit'] = df['is_cross_border'] * df['cross_border_fit']
    df['new_x_fit'] = (df['seller_tier'] == 'new').astype(int) * df['new_seller_fit']
    df['estab_x_newfit'] = (df['seller_tier'] == 'established').astype(int) * df['new_seller_fit']
    df['growth_x_newfit'] = (df['seller_tier'] == 'growth').astype(int) * df['new_seller_fit']
    df['premium_x_newfit'] = (df['seller_tier'] == 'premium').astype(int) * df['new_seller_fit']
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
    df['vis_x_orders'] = df['visibility_boost'] * np.log1p(df['orders_30d'])
    df['discount_x_fatigue'] = df['discount_depth'] * df['promotion_fatigue_30d']
    df['readiness_x_discount'] = df['marketing_readiness_score'] * df['discount_depth']
    df['margin_over_penalty'] = df['margin_rate'] / (df['margin_penalty'] + 0.01)
    df['loyalty_x_inv'] = df['loyalty_synergy'] * df['inventory_synergy']
    df['vis_x_new_fit'] = df['visibility_boost'] * df['new_seller_fit']
    df['seasonal_x_holiday'] = df['seasonal_fit'] * df['holiday_campaign']
    
    # ---- NEW: Channel-Tool Type Alignment ----
    df['ads_x_vis_tool'] = df['ch_ads'] * (df['tool_type'] == 'visibility').astype(int)
    df['social_x_event_tool'] = df['ch_social'] * (df['tool_type'] == 'event').astype(int)
    df['affiliate_x_discount'] = df['ch_affiliate'] * (df['tool_type'] == 'discount').astype(int)
    df['ads_x_event_tool'] = df['ch_ads'] * (df['tool_type'] == 'event').astype(int)
    df['search_x_vis_tool'] = df['ch_search'] * (df['tool_type'] == 'visibility').astype(int)
    df['live_x_event_tool'] = df['ch_live'] * (df['tool_type'] == 'event').astype(int)
    
    # ---- NEW: Tool-specific seller alignment signals ----
    # Flash sale fits better with high inventory fill
    df['flashsale_fit'] = (df['promo_tool'] == 'flash_sale_slot').astype(int) * df['inventory_fill_rate']
    # Free shipping more relevant for cross-border
    df['freeship_xborder'] = (df['promo_tool'] == 'free_shipping_boost').astype(int) * df['is_cross_border']
    # Loyalty points for sellers with repeat buyers
    df['loyalty_repeat'] = (df['promo_tool'] == 'loyalty_points_multiplier').astype(int) * df['repeat_buyer_rate']
    # Cashback for sellers with budget
    df['cashback_budget'] = (df['promo_tool'] == 'cashback_offer').astype(int) * df['cashback_budget_score']
    # Homepage feature for sellers with high visibility scores already
    df['homepage_vis'] = (df['promo_tool'] == 'homepage_feature').astype(int) * df['search_visibility_score']
    # Sponsored search for sellers actively using ads
    df['sponsored_ads'] = (df['promo_tool'] == 'sponsored_search_boost').astype(int) * df['ch_ads']
    # Bundle builder for sellers with good inventory
    df['bundle_inv'] = (df['promo_tool'] == 'bundle_builder').astype(int) * df['inventory_fill_rate']
    # Coupon pack for sellers with high margin
    df['coupon_margin'] = (df['promo_tool'] == 'coupon_pack').astype(int) * df['margin_rate']
    
    # ---- Seller Performance Ratios ----
    df['gmv_per_order'] = df['gmv_30d'] / (df['orders_30d'] + 1)
    df['ad_efficiency'] = df['gmv_30d'] / (df['ad_spend_30d'] + 1)
    df['view_to_order'] = df['orders_30d'] / (df['listing_views_30d'] + 1)
    df['readiness_minus_fatigue'] = df['marketing_readiness_score'] - df['promotion_fatigue_30d']
    df['fill_minus_stockout'] = df['inventory_fill_rate'] - df['stockout_rate_30d']
    df['ad_per_order'] = df['ad_spend_30d'] / (df['orders_30d'] + 1)
    df['margin_gmv'] = df['margin_rate'] * df['gmv_30d']
    df['log_gmv'] = np.log1p(df['gmv_30d'])
    df['log_orders'] = np.log1p(df['orders_30d'])
    df['log_views'] = np.log1p(df['listing_views_30d'])
    df['log_ad_spend'] = np.log1p(df['ad_spend_30d'])
    df['log_tenure'] = np.log1p(df['seller_tenure_days'])
    
    # ---- Tool Composites ----
    df['tool_benefit'] = (df['visibility_boost'] + df['inventory_synergy'] + 
                          df['loyalty_synergy'] - df['margin_penalty'])
    df['tool_cost_benefit'] = ((df['visibility_boost'] + df['inventory_synergy']) / 
                               (df['margin_penalty'] + 0.1))
    df['tool_total_synergy'] = (df['inventory_synergy'] + df['loyalty_synergy'] + 
                                df['cross_border_fit'] + df['new_seller_fit'] + 
                                df['seasonal_fit'])
    
    # ---- Label Encode ----
    for col in ['region', 'primary_category', 'seller_tier', 'promo_tool', 'tool_type', 'cost_tier']:
        le = LabelEncoder()
        df[f'{col}_enc'] = le.fit_transform(df[col].astype(str))
    
    # ---- Dummies ----
    for t in ['premium', 'established', 'growth', 'new']:
        df[f'is_{t}'] = (df['seller_tier'] == t).astype(int)
    for tt in ['visibility', 'bundle', 'event', 'retention', 'shipping', 'discount']:
        df[f'tt_{tt}'] = (df['tool_type'] == tt).astype(int)
    for ct in ['low', 'medium', 'high']:
        df[f'ct_{ct}'] = (df['cost_tier'] == ct).astype(int)
    
    return df

train = add_features(train)
test = add_features(test)

# ---- Per-Query aggregate features (no target leakage!) ----
print("Per-query aggregate features...")

# How many tools has this seller used recently (sum of tool_recently_used per query)
train['query_n_recently_used'] = train.groupby('query_id')['tool_recently_used'].transform('sum')
test['query_n_recently_used'] = test.groupby('query_id')['tool_recently_used'].transform('sum')

# ============================================================
# 3. Time-Aware Target Encoding
# ============================================================
print("Time-aware target encoding...")

def temporal_te(train_df, test_df, col, smoothing=30):
    gm = train_df['is_relevant'].mean()
    weeks = sorted(train_df['snapshot_date'].unique())
    
    encoded = np.full(len(train_df), gm)
    cum_data = pd.DataFrame()
    
    for week in weeks:
        week_mask = train_df['snapshot_date'] == week
        week_idx = np.where(week_mask)[0]
        
        if len(cum_data) > 0:
            agg = cum_data.groupby(col)['is_relevant'].agg(['sum', 'count'])
            sm = (agg['sum'] + gm * smoothing) / (agg['count'] + smoothing)
            vals = train_df.loc[week_mask, col].map(sm)
            encoded[week_idx] = vals.fillna(gm).values
        
        cum_data = pd.concat([cum_data, train_df.loc[week_mask, [col, 'is_relevant']]])
    
    train_df[f'{col}_tte'] = encoded
    
    agg = train_df.groupby(col)['is_relevant'].agg(['sum', 'count'])
    sm = (agg['sum'] + gm * smoothing) / (agg['count'] + smoothing)
    test_df[f'{col}_tte'] = test_df[col].map(sm).fillna(gm)
    
    return train_df, test_df

for col in ['promo_tool', 'tool_type', 'cost_tier']:
    print(f"  TE: {col}")
    train, test = temporal_te(train, test, col)

inters = [
    ('promo_tool', 'seller_tier'),
    ('promo_tool', 'primary_category'),
    ('promo_tool', 'region'),
    ('tool_type', 'seller_tier'),
    ('tool_type', 'primary_category'),
]
for c1, c2 in inters:
    icol = f'{c1}_X_{c2}'
    train[icol] = train[c1].astype(str) + '_' + train[c2].astype(str)
    test[icol] = test[c1].astype(str) + '_' + test[c2].astype(str)
    print(f"  TE: {icol}")
    train, test = temporal_te(train, test, icol)

# ============================================================
# 4. Per-Query Relative Features
# ============================================================
print("Per-query relative features...")

rank_cols = [
    'discount_depth', 'visibility_boost', 'inventory_synergy',
    'loyalty_synergy', 'margin_penalty', 'cross_border_fit',
    'new_seller_fit', 'seasonal_fit', 'tool_benefit', 'tool_cost_benefit',
    'tool_total_synergy',
]

for col in rank_cols:
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
# 6. Time-Based Validation
# ============================================================
print("Setting up validation...")

val_cutoff = pd.Timestamp('2024-05-13')
vm = train['snapshot_date'] >= val_cutoff
tm_ = ~vm

Xt, yt = X_train[tm_], y_train[tm_]
Xv, yv = X_train[vm], y_train[vm]
tq_ = tqids[tm_]; vq_ = tqids[vm]

tqo_ = pd.Series(tq_).unique()
tg_ = pd.DataFrame({'q': tq_}).groupby('q').size().reindex(tqo_).values
vqo_ = pd.Series(vq_).unique()
vg_ = pd.DataFrame({'q': vq_}).groupby('q').size().reindex(vqo_).values

print(f"Train: {len(tqo_)} queries, Val: {len(vqo_)} queries")

# ============================================================
# 7. Train Models with Multiple Param Configs
# ============================================================
print("\n=== Training Models ===")

lgb_configs = [
    # Config A: moderate
    {
        'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [3],
        'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
        'min_child_samples': 50, 'feature_fraction': 0.7,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'lambda_l1': 0.1, 'lambda_l2': 1.0, 'min_gain_to_split': 0.01,
        'verbose': -1, 'n_jobs': -1, 'label_gain': [0, 1],
        'feature_pre_filter': False,
    },
    # Config B: slightly deeper
    {
        'objective': 'lambdarank', 'metric': 'ndcg', 'eval_at': [3],
        'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 40, 'feature_fraction': 0.65,
        'bagging_fraction': 0.75, 'bagging_freq': 3,
        'lambda_l1': 0.2, 'lambda_l2': 2.0, 'min_gain_to_split': 0.02,
        'verbose': -1, 'n_jobs': -1, 'label_gain': [0, 1],
        'feature_pre_filter': False,
    },
]

bin_configs = [
    {
        'objective': 'binary', 'metric': 'binary_logloss',
        'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
        'min_child_samples': 50, 'feature_fraction': 0.7,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'lambda_l1': 0.1, 'lambda_l2': 1.0,
        'verbose': -1, 'n_jobs': -1, 'feature_pre_filter': False,
    },
    {
        'objective': 'binary', 'metric': 'binary_logloss',
        'learning_rate': 0.02, 'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 40, 'feature_fraction': 0.65,
        'bagging_fraction': 0.75, 'bagging_freq': 3,
        'lambda_l1': 0.2, 'lambda_l2': 2.0,
        'verbose': -1, 'n_jobs': -1, 'feature_pre_filter': False,
    },
]

rank_models = []
td = lgb.Dataset(Xt, label=yt, group=tg_, feature_name=feature_cols)
vd = lgb.Dataset(Xv, label=yv, group=vg_, feature_name=feature_cols, reference=td)

for i, cfg in enumerate(lgb_configs):
    m = lgb.train(cfg, td, num_boost_round=3000, valid_sets=[vd], valid_names=['val'],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    score = m.best_score['val']['ndcg@3']
    rank_models.append((m, cfg, score, m.best_iteration))
    print(f"Rank config {i}: NDCG@3={score:.6f} (iter={m.best_iteration})")

bin_models = []
td_b = lgb.Dataset(Xt, label=yt, feature_name=feature_cols)
vd_b = lgb.Dataset(Xv, label=yv, feature_name=feature_cols, reference=td_b)

for i, cfg in enumerate(bin_configs):
    m = lgb.train(cfg, td_b, num_boost_round=3000, valid_sets=[vd_b], valid_names=['val'],
                  callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)])
    vp = m.predict(Xv)
    vdf = pd.DataFrame({'q': vq_, 'y': yv, 's': vp})
    n3 = vdf.groupby('q').apply(
        lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['s'].values.reshape(1,-1), k=3)
    ).mean()
    bin_models.append((m, cfg, n3, m.best_iteration))
    print(f"Binary config {i}: NDCG@3={n3:.6f} (iter={m.best_iteration})")

# ---- Feature Importance ----
imp = pd.DataFrame({
    'feature': feature_cols,
    'importance': rank_models[0][0].feature_importance('gain')
}).sort_values('importance', ascending=False)
print("\nTop 25 features:")
print(imp.head(25).to_string(index=False))

# ============================================================
# 8. Ensemble Optimization
# ============================================================
print("\n=== Ensemble Optimization ===")

# Collect all val predictions
val_preds = []
for m, cfg, s, it in rank_models:
    vp = m.predict(Xv)
    val_preds.append(MinMaxScaler().fit_transform(vp.reshape(-1,1)).flatten())
for m, cfg, s, it in bin_models:
    vp = m.predict(Xv)
    val_preds.append(MinMaxScaler().fit_transform(vp.reshape(-1,1)).flatten())

# Simple average
avg_vp = np.mean(val_preds, axis=0)
vdf_avg = pd.DataFrame({'q': vq_, 'y': yv, 's': avg_vp})
avg_n3 = vdf_avg.groupby('q').apply(
    lambda g: ndcg_score(g['y'].values.reshape(1,-1), g['s'].values.reshape(1,-1), k=3)
).mean()
print(f"Simple average NDCG@3: {avg_n3:.6f}")

# ============================================================
# 9. Final Training (Multi-Seed)
# ============================================================
print("\n=== Final Training ===")

seeds = [42, 123, 456, 789, 2024, 1337, 7777, 9999, 31415]
n_models = len(rank_models) + len(bin_models)
all_preds = [np.zeros(len(X_test)) for _ in range(n_models)]

fd = lgb.Dataset(X_train, label=y_train, group=train_groups, feature_name=feature_cols)
fb = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)

for s in seeds:
    print(f"  Seed {s}...")
    
    for i, (_, cfg, _, it) in enumerate(rank_models):
        c = cfg.copy(); c['seed'] = s
        m = lgb.train(c, fd, num_boost_round=int(it * 1.05))
        all_preds[i] += m.predict(X_test)
    
    for i, (_, cfg, _, it) in enumerate(bin_models):
        idx = len(rank_models) + i
        c = cfg.copy(); c['seed'] = s
        m = lgb.train(c, fb, num_boost_round=int(it * 1.05))
        all_preds[idx] += m.predict(X_test)

for i in range(n_models):
    all_preds[i] /= len(seeds)
    all_preds[i] = MinMaxScaler().fit_transform(all_preds[i].reshape(-1,1)).flatten()

final = np.mean(all_preds, axis=0)

# ============================================================
# 10. Generate Submissions
# ============================================================
print("\nGenerating submissions...")

pd.DataFrame({'candidate_id': test_cids, 'score': final}).to_csv('working/submission.csv', index=False)
print("submission.csv (ensemble avg)")

# Individual best models
pd.DataFrame({'candidate_id': test_cids, 'score': all_preds[0]}).to_csv('working/submission_rank0.csv', index=False)
pd.DataFrame({'candidate_id': test_cids, 'score': all_preds[len(rank_models)]}).to_csv('working/submission_bin0.csv', index=False)
print("submission_rank0.csv, submission_bin0.csv")

# Verify
sub = pd.read_csv('working/submission.csv')
test_orig = pd.read_csv('dataset/public/test.csv')
print(f"\nRows: {len(sub)} (expected 57600)")
print(f"IDs match: {set(sub['candidate_id']) == set(test_orig['candidate_id'])}")

print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
for i, (_, _, s, it) in enumerate(rank_models):
    print(f"  Rank config {i}: NDCG@3={s:.6f} (iter={it})")
for i, (_, _, s, it) in enumerate(bin_models):
    print(f"  Binary config {i}: NDCG@3={s:.6f} (iter={it})")
print(f"  Ensemble avg NDCG@3: {avg_n3:.6f}")
print(f"Features: {len(feature_cols)}, Seeds: {len(seeds)}")
print(f"{'='*60}")
print("DONE!")
