import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
import lightgbm as lgb

# ================== LOAD DATA ==================
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

target = "target"  # change if needed

X = train.drop(columns=[target])
y = train[target]

# ================== BASIC PREPROCESS ==================
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# fill missing
for col in num_cols:
    X[col].fillna(X[col].median(), inplace=True)
    test[col].fillna(test[col].median(), inplace=True)

# label encode categoricals
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    test[col] = le.transform(test[col].astype(str))

# ================== FEATURE ENGINEERING ==================
X['missing_count'] = X.isnull().sum(axis=1)
test['missing_count'] = test.isnull().sum(axis=1)

# example ratio (edit based on your cols)
if 'cu_total_approved' in X.columns and 'cu_total_enrolled' in X.columns:
    X['approval_rate'] = X['cu_total_approved'] / (X['cu_total_enrolled'] + 1)
    test['approval_rate'] = test['cu_total_approved'] / (test['cu_total_enrolled'] + 1)

# ================== CV SETUP ==================
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros((len(X), len(np.unique(y))))
test_preds = np.zeros((len(test), len(np.unique(y))))

# ================== TRAINING ==================
for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"\nFold {fold+1}")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # ---- CATBOOST ----
    cat_model = CatBoostClassifier(
        iterations=2000,
        learning_rate=0.03,
        depth=6,
        loss_function='MultiClass',
        eval_metric='MultiClass',
        verbose=200
    )

    cat_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=200
    )

    # ---- LIGHTGBM ----
    lgb_model = lgb.LGBMClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=64,
        max_depth=6
    )

    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(200)]
    )

    # ---- XGBOOST ----
    xgb_model = XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        tree_method='hist'
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=200,
        verbose=False
    )

    # ---- PREDICTIONS ----
    cat_val = cat_model.predict_proba(X_val)
    lgb_val = lgb_model.predict_proba(X_val)
    xgb_val = xgb_model.predict_proba(X_val)

    val_pred = (0.5 * cat_val + 0.3 * lgb_val + 0.2 * xgb_val)
    oof_preds[val_idx] = val_pred

    # test preds
    cat_test = cat_model.predict_proba(test)
    lgb_test = lgb_model.predict_proba(test)
    xgb_test = xgb_model.predict_proba(test)

    test_preds += (0.5 * cat_test + 0.3 * lgb_test + 0.2 * xgb_test) / skf.n_splits

# ================== FINAL EVAL ==================
final_preds = np.argmax(oof_preds, axis=1)
print("CV Accuracy:", accuracy_score(y, final_preds))

# ================== SAVE SUBMISSION ==================
submission = pd.DataFrame({
    "id": test.index,
    "target": np.argmax(test_preds, axis=1)
})

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")
