import logging
import re
import pandas as pd
import numpy as np
from serverlex.preprocessing.clean import Clean

from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor

logger=logging.getLogger("Pipeline")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)


class MLPipeline:
    def __init__(self, data=None):
        self.data=None
        self.cleaner=None
        self.target=None
        self.feature_columns=[]
        self.num_fill={}
        self.cat_fill={}
        self.cat_categories={}
        self.scaler_mean={}
        self.scaler_std={}
        self.scaler_min={}
        self.scaler_max={}
        self.scaler_median={}
        self.scaler_iqr={}
        self.date_col=None
        self.id_cols=[]
        self.custom_models={}
        self.X_train=None
        self.y_train=None

        if data is not None:
            self.load_data(data)

    def load_data(self, data):
        if isinstance(data, str):
            self.cleaner=Clean(data)
            self.cleaner.load()
            self.data=self.cleaner.data
        elif isinstance(data, Clean):
            self.cleaner=data
            self.data=self.cleaner.data
        elif isinstance(data, pd.DataFrame):
            self.cleaner=Clean(data)
            self.data=self.cleaner.data

    # ==========================
    # 1. FIT (Learn & Save on self)
    # ==========================
    def fit(self, df, target=None, encode='onehot', scale='standard', data_time=None, numeric_string_cols=None, drop_id_cols=True, id_cols=None, **kwargs):
        try:
            valid_scales = ['standard', 'minmax', 'robust', None]
            if scale not in valid_scales:
                raise ValueError(f"Invalid scale method '{scale}'. Supported options: {valid_scales}")

            valid_encodes = ['onehot', 'label', None]
            if encode not in valid_encodes:
                raise ValueError(f"Invalid encode method '{encode}'. Supported options: {valid_encodes}")

            self.load_data(df)
            self.target=target.strip().lower() if target else None
            self.encode=encode
            self.scale=scale
            self.numeric_string_cols=[c.strip().lower().replace(' ', '_') for c in (numeric_string_cols or [])]
            self.drop_id_cols=drop_id_cols

            logger.info("Fitting pipeline on %d rows (target=%s, encode=%s, scale=%s)", len(self.data), self.target, self.encode, self.scale)

            # 0. Auto-detect object columns that are predominantly numeric (≥70% parseable).
            #    This handles columns read as strings from dirty CSVs (e.g. "9.6 GB", "50 kg", "0,0")
            #    so they are coerced to numeric rather than one-hot encoded as categoricals.
            _UNIT_PAT = r'(?i)\s*(gb|mb|kb|tb|pb|yrs?|years?|months?|days?|hours?|hrs?|mins?|sec|seconds?|%|km|miles?|kg|lbs?)\s*$'
            _SYM_PAT = r'[₹\$,/€£¥\s]'
            _auto_numeric = set(numeric_string_cols or [])
            for col in self.data.select_dtypes(include=['object', 'string']).columns:
                if self.target and col.strip().lower() == self.target:
                    continue
                sample = self.data[col].dropna().astype(str).str.strip()
                if sample.empty or len(sample) < 5:
                    continue
                cleaned_sample = (
                    sample
                    .str.replace(_UNIT_PAT, '', regex=True)
                    .str.replace(_SYM_PAT, '', regex=True)
                    .str.replace(',', '.', regex=False)
                    .str.strip()
                )
                parse_rate = pd.to_numeric(cleaned_sample, errors='coerce').notna().mean()
                if parse_rate >= 0.70:
                    _auto_numeric.add(col)
            if _auto_numeric:
                extra = sorted(_auto_numeric - set(numeric_string_cols or []))
                if extra:
                    logger.info("Auto-detected numeric-majority string columns (will coerce): %s", extra)
            numeric_string_cols = list(_auto_numeric) if _auto_numeric else numeric_string_cols

            # 1. Clean the training data using your Clean class with target protection
            clean_kwargs = kwargs.copy()
            if self.target:
                exclude = clean_kwargs.get("exclude_outliers", [])
                if isinstance(exclude, (list, tuple, set)):
                    exclude = list(exclude)
                else:
                    exclude = [exclude]
                if self.target not in exclude:
                    exclude.append(self.target)
                clean_kwargs["exclude_outliers"] = exclude
                if "corr_target" not in clean_kwargs:
                    clean_kwargs["corr_target"] = self.target

            self.cleaner.total_clean(data_time=data_time, numeric_string_cols=numeric_string_cols, **clean_kwargs)
            self.data=self.cleaner.data
            self.cleaner.separate()

            # 2. Extract datetime features if provided
            if data_time:
                col=data_time[0].strip().lower().replace(' ', '_')
                if col in self.data.columns:
                    self.date_col=col
                    dt=pd.to_datetime(self.data[col], format='mixed', errors='coerce')
                    self.data[f'{col}_year']=dt.dt.year
                    self.data[f'{col}_month']=dt.dt.month
                    self.data[f'{col}_day']=dt.dt.day
                    self.data[f'{col}_dayofweek']=dt.dt.dayofweek
                    self.data=self.data.drop(columns=[col])

            self.cleaner.data=self.data
            self.cleaner.separate()

            # 3. Detect and skip ID columns
            #    DataGuard runs a lightweight uniqueness scan on the *cleaned* data so
            #    column names are already normalized (e.g. "Cust ID" → "cust_id").
            _id_cols_set = set()
            if id_cols is not None:
                # Caller explicitly supplied the list — normalize and trust it.
                _id_cols_set = {c.strip().lower().replace(' ', '_') for c in id_cols}
            elif drop_id_cols:
                try:
                    from serverlex.diagnostics.dataguard import DataGuard as _DG
                    _guard = _DG(self.data)
                    _guard.check_id_columns()
                    _id_cols_set = set(getattr(_guard, 'id_cols', []))
                    if _id_cols_set:
                        logger.warning(
                            "DataGuard detected ID column(s) — skipping from encoding and feature matrix: %s. "
                            "Pass drop_id_cols=False to override.",
                            sorted(_id_cols_set)
                        )
                except Exception:
                    pass  # DataGuard failure is non-fatal; proceed without exclusion
            self.id_cols = sorted(list(_id_cols_set))

            # 4. Memorize missing value fill values directly on self (excluding ID columns & target)
            for col in self.cleaner.numerical_columns.columns:
                if col != self.target and col not in _id_cols_set:
                    self.num_fill[col]=float(self.data[col].mean()) if not self.data[col].dropna().empty else 0.0

            for col in self.cleaner.categorical_columns.columns:
                if col != self.target and col not in _id_cols_set:
                    cleaned_series = self.data[col].dropna().astype(str).str.lower().str.replace(' ', '')
                    mode_val = cleaned_series.mode()
                    self.cat_fill[col] = mode_val[0] if not mode_val.empty else 'unknown'

            # 5. Encode categoricals and memorize exact categories
            train_X=self.data.copy()
            drop_from_train = set(_id_cols_set)
            if self.target and self.target in train_X.columns:
                drop_from_train.add(self.target)
            train_X=train_X.drop(columns=[c for c in drop_from_train if c in train_X.columns])

            cat_cols=[c for c in self.cleaner.categorical_columns.columns
                      if c != self.target and c in train_X.columns and c not in _id_cols_set]
            self.cat_categories = {}
            for col in cat_cols:
                # Fill missing before string casting (Bug 3 fix)
                train_X[col] = train_X[col].fillna(self.cat_fill[col]).astype(str).str.lower().str.replace(' ', '')
                cats = sorted(list(train_X[col].unique()))
                self.cat_categories[col] = cats
                train_X[col] = pd.Categorical(train_X[col], categories=cats)

            if self.encode=='onehot' and cat_cols:
                train_X=pd.get_dummies(train_X, columns=cat_cols, drop_first=True, dtype=int)
            elif self.encode=='label' and cat_cols:
                for col in cat_cols:
                    train_X[col]=train_X[col].cat.codes

            # Save exact list of final columns for future test data
            self.feature_columns=train_X.columns.tolist()

            # 5. Memorize scaling numbers on self
            num_cols_to_scale = [c for c in self.cleaner.numerical_columns.columns if c in self.feature_columns]
            if self.scale=='standard':
                for col in num_cols_to_scale:
                    self.scaler_mean[col]=float(train_X[col].mean())
                    std_val=float(train_X[col].std())
                    self.scaler_std[col]=std_val if std_val != 0 else 1.0

            elif self.scale=='minmax':
                for col in num_cols_to_scale:
                    min_val = float(train_X[col].min())
                    max_val = float(train_X[col].max())
                    self.scaler_min[col] = min_val
                    self.scaler_max[col] = max_val if max_val != min_val else min_val + 1.0

            elif self.scale=='robust':
                for col in num_cols_to_scale:
                    median_val = float(train_X[col].median())
                    q75 = float(train_X[col].quantile(0.75))
                    q25 = float(train_X[col].quantile(0.25))
                    iqr = q75 - q25
                    self.scaler_median[col] = median_val
                    self.scaler_iqr[col] = iqr if iqr != 0 else 1.0

            # 6. Populate self.X_train and self.y_train so evaluate_baseline_models() works immediately
            res = self.transform(self.data)
            if isinstance(res, tuple):
                self.X_train, self.y_train = res[0], res[1]
            else:
                self.X_train = res
                self.y_train = None

            logger.info("Fit complete. Feature count: %d", len(self.feature_columns))
            return self
        except Exception as e:
            logger.error("Error during fit: %s", e)
            raise

    # ==========================
    # 2. TRANSFORM (Uses self variables)
    # ==========================
    def transform(self, df):
        try:
            out=df.copy()

            # 1. Clean column names
            out.columns=out.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)

            # Clean numeric string columns using saved self.numeric_string_cols
            _UNIT_PAT2 = r'(?i)\s*(gb|mb|kb|tb|pb|yrs?|years?|months?|days?|hours?|hrs?|mins?|sec|seconds?|%|km|miles?|kg|lbs?)\s*$'
            _SYM_PAT2 = r'[₹\$,/€£¥]'
            if hasattr(self, 'numeric_string_cols') and self.numeric_string_cols:
                for col in self.numeric_string_cols:
                    if col in out.columns:
                        cleaned=out[col].astype(str).str.replace(_SYM_PAT2, '', regex=True).str.replace(_UNIT_PAT2, '', regex=True).str.replace(',', '.', regex=False).str.strip()
                        out[col]=pd.to_numeric(cleaned, errors='coerce')

            # Also coerce any num_fill column that is still string/object dtype
            # (handles auto-detected columns stored after column name cleaning)
            for col in list(self.num_fill.keys()):
                if col in out.columns and (out[col].dtype == object or str(out[col].dtype).startswith('str')):
                    cleaned = (
                        out[col].astype(str)
                        .str.replace(_UNIT_PAT2, '', regex=True)
                        .str.replace(_SYM_PAT2, '', regex=True)
                        .str.replace(',', '.', regex=False)
                        .str.strip()
                    )
                    out[col] = pd.to_numeric(cleaned, errors='coerce')

            # 2. Extract datetime features using saved date_col
            if self.date_col and self.date_col in out.columns:
                dt=pd.to_datetime(out[self.date_col], format='mixed', errors='coerce')
                out[f'{self.date_col}_year']=dt.dt.year
                out[f'{self.date_col}_month']=dt.dt.month
                out[f'{self.date_col}_day']=dt.dt.day
                out[f'{self.date_col}_dayofweek']=dt.dt.dayofweek
                out=out.drop(columns=[self.date_col])

            # 2.5 Drop ID columns if present
            if hasattr(self, 'id_cols') and self.id_cols:
                out=out.drop(columns=[c for c in self.id_cols if c in out.columns])

            # 3. Fill missing using memorized self.num_fill and self.cat_fill
            for col, val in self.num_fill.items():
                if col in out.columns:
                    out[col]=out[col].fillna(val)

            for col, val in self.cat_fill.items():
                if col in out.columns:
                    # Fill NA before converting to string (Bug 3 fix)
                    out[col]=out[col].fillna(val)
                    out[col]=out[col].astype(str).str.lower().str.replace(' ', '')

            # 4. Check target
            has_target=self.target and self.target in out.columns
            target_series=out[self.target] if has_target else None

            # 5. Encode categoricals using strictly memorized categories (Bug 1 fix)
            cat_cols=[c for c in self.cat_categories.keys() if c in out.columns]
            if cat_cols:
                for col in cat_cols:
                    # Map novel categories to NaN so they drop/zero cleanly
                    mask = out[col].isin(self.cat_categories[col])
                    out.loc[~mask, col] = np.nan
                    out[col] = pd.Categorical(out[col], categories=self.cat_categories[col])

                if self.encode=='onehot':
                    out=pd.get_dummies(out, columns=cat_cols, drop_first=True, dtype=int)
                elif self.encode=='label':
                    for col in cat_cols:
                        out[col]=out[col].cat.codes

            # 6. Reindex columns so test columns match self.feature_columns exactly
            X=out.reindex(columns=self.feature_columns, fill_value=0)

            # 7. Scale using memorized statistics
            if self.scale=='standard':
                for col in self.scaler_mean.keys():
                    if col in X.columns:
                        X[col]=(X[col] - self.scaler_mean[col]) / self.scaler_std[col]
            elif self.scale=='minmax':
                for col in self.scaler_min.keys():
                    if col in X.columns:
                        X[col]=(X[col] - self.scaler_min[col]) / (self.scaler_max[col] - self.scaler_min[col])
            elif self.scale=='robust':
                for col in self.scaler_median.keys():
                    if col in X.columns:
                        X[col]=(X[col] - self.scaler_median[col]) / self.scaler_iqr[col]

            if has_target:
                return X, target_series
            return X
        except Exception as e:
            logger.error("Error during transform: %s", e)
            raise

    def fit_transform(self, df, **kwargs):
        self.fit(df, **kwargs)
        if self.y_train is not None:
            return self.X_train, self.y_train
        return self.X_train

    # ==========================
    # 3. Custom Models & Evaluation
    # ==========================
    def add_model(self, name, model):
        self.custom_models[name]=model
        logger.info("Registered custom model '%s'", name)
        return self

    def evaluate_baseline_models(self, X=None, y=None, cv=5, verbose=False):
        try:
            if X is None and y is None:
                if hasattr(self, 'X_train') and self.X_train is not None and hasattr(self, 'y_train') and self.y_train is not None:
                    X, y = self.X_train, self.y_train
                else:
                    raise ValueError("No data provided and pipeline has not been fit_transformed. Pass (X, y) or run fit_transform() first.")

            if not isinstance(y, pd.Series):
                y=pd.Series(y)

            # Trustworthy classification detection: check dtype or integer discrete
            is_classification = (
                y.dtype == 'object'
                or str(y.dtype).startswith('cat')
                or str(y.dtype).startswith('bool')
                or (pd.api.types.is_integer_dtype(y) and y.nunique() <= 20)
            )
            task="Classification" if is_classification else "Regression"

            results=[]

            if is_classification:
                kf=StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
                models={
                    "Logistic Regression": LogisticRegression(max_iter=1000),
                    "Decision Tree": DecisionTreeClassifier(random_state=42),
                    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingClassifier(random_state=42)
                }
                models.update(self.custom_models)

                if verbose:
                    logger.info("Evaluating %d baseline models (task=%s, cv=%d)", len(models), task, cv)

                for name, model in models.items():
                    acc_scores=cross_val_score(model, X, y, cv=kf, scoring='accuracy')
                    f1_scores=cross_val_score(model, X, y, cv=kf, scoring='f1_weighted')
                    results.append({
                        "Model": name,
                        "Accuracy": round(float(acc_scores.mean()), 4),
                        "F1_Weighted": round(float(f1_scores.mean()), 4)
                    })
            else:
                kf=KFold(n_splits=cv, shuffle=True, random_state=42)
                models={
                    "Ridge Regression": Ridge(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
                }
                models.update(self.custom_models)

                if verbose:
                    logger.info("Evaluating %d baseline models (task=%s, cv=%d)", len(models), task, cv)

                for name, model in models.items():
                    r2_scores=cross_val_score(model, X, y, cv=kf, scoring='r2')
                    rmse_scores=-cross_val_score(model, X, y, cv=kf, scoring='neg_root_mean_squared_error')
                    results.append({
                        "Model": name,
                        "R2": round(float(r2_scores.mean()), 4),
                        "RMSE": round(float(rmse_scores.mean()), 4)
                    })

            res_df=pd.DataFrame(results).sort_values(by=list(results[0].keys())[1], ascending=False)
            if verbose:
                logger.info("Baseline leaderboard:\n%s", res_df.to_string(index=False))
            return res_df
        except Exception as e:
            logger.error("Error evaluating models: %s", e)
            raise
