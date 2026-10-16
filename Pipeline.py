import logging
import pandas as pd
import numpy as np
from Clean import Clean

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
        self.scaler_mean={}
        self.scaler_std={}
        self.date_col=None
        self.custom_models={}

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
            self.cleaner=Clean.__new__(Clean)
            self.cleaner.data=data.copy()
            self.cleaner.path=None
            self.data=data

    # ==========================
    # 1. FIT (Learn & Save on self)
    # ==========================
    def fit(self, df, target=None, encode='onehot', scale='standard', data_time=None, numeric_string_cols=None, **kwargs):
        try:
            self.load_data(df)
            self.target=target.strip().lower() if target else None
            self.encode=encode
            self.scale=scale
            self.numeric_string_cols=[c.strip().lower().replace(' ', '_') for c in (numeric_string_cols or [])]

            logger.info("Fitting pipeline on %d rows (target=%s, encode=%s, scale=%s)", len(self.data), self.target, self.encode, self.scale)

            # 1. Clean the training data using your Clean class
            self.cleaner.total_clean(data_time=data_time, numeric_string_cols=numeric_string_cols, **kwargs)
            self.data=self.cleaner.data
            self.cleaner.seperate()

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
            self.cleaner.seperate()

            # 3. Memorize missing value fill values directly on self
            for col in self.cleaner.numerical_columns.columns:
                if col != self.target:
                    self.num_fill[col]=self.data[col].mean()

            for col in self.cleaner.categrorical_columns.columns:
                if col != self.target:
                    mode_val=self.data[col].mode()
                    self.cat_fill[col]=mode_val[0] if not mode_val.empty else 'unknown'

            # 4. Encode categoricals to find final feature columns
            train_X=self.data.copy()
            if self.target and self.target in train_X.columns:
                train_X=train_X.drop(columns=[self.target])

            cat_cols=[c for c in self.cleaner.categrorical_columns.columns if c != self.target and c in train_X.columns]
            if self.encode=='onehot' and cat_cols:
                train_X=pd.get_dummies(train_X, columns=cat_cols, drop_first=True, dtype=int)
            elif self.encode=='label' and cat_cols:
                for col in cat_cols:
                    train_X[col]=train_X[col].astype('category').cat.codes

            # Save exact list of final columns for future test data
            self.feature_columns=train_X.columns.tolist()

            # 5. Memorize scaling numbers on self
            if self.scale=='standard':
                for col in self.cleaner.numerical_columns.columns:
                    if col in self.feature_columns:
                        self.scaler_mean[col]=train_X[col].mean()
                        std_val=train_X[col].std()
                        self.scaler_std[col]=std_val if std_val != 0 else 1.0

            logger.info("Fit complete. Feature count: %d", len(self.feature_columns))
            return self
        except Exception as e:
            logger.error("Error during fit: %s", e)
            return self

    # ==========================
    # 2. TRANSFORM (Uses self variables)
    # ==========================
    def transform(self, df):
        try:
            out=df.copy()

            # 1. Clean column names
            out.columns=out.columns.str.strip().str.lower().str.replace(' ', '_').str.replace(r'[^\w_]', '', regex=True)

            # Clean numeric string columns using saved self.numeric_string_cols
            if hasattr(self, 'numeric_string_cols'):
                for col in self.numeric_string_cols:
                    if col in out.columns:
                        cleaned=out[col].astype(str).str.replace(r'[\$,]', '', regex=True)
                        out[col]=pd.to_numeric(cleaned, errors='coerce')

            # 2. Extract datetime features using saved date_col
            if self.date_col and self.date_col in out.columns:
                dt=pd.to_datetime(out[self.date_col], format='mixed', errors='coerce')
                out[f'{self.date_col}_year']=dt.dt.year
                out[f'{self.date_col}_month']=dt.dt.month
                out[f'{self.date_col}_day']=dt.dt.day
                out[f'{self.date_col}_dayofweek']=dt.dt.dayofweek
                out=out.drop(columns=[self.date_col])

            # 3. Fill missing using memorized self.num_fill and self.cat_fill
            for col, val in self.num_fill.items():
                if col in out.columns:
                    out[col]=out[col].fillna(val)

            for col, val in self.cat_fill.items():
                if col in out.columns:
                    out[col]=out[col].astype(str).str.lower().str.replace(' ', '')
                    out[col]=out[col].fillna(val)

            # 4. Check target
            has_target=self.target and self.target in out.columns
            target_series=out[self.target] if has_target else None

            # 5. Encode categoricals
            cat_cols=[c for c in self.cat_fill.keys() if c in out.columns]
            if self.encode=='onehot' and cat_cols:
                out=pd.get_dummies(out, columns=cat_cols, drop_first=True, dtype=int)
            elif self.encode=='label' and cat_cols:
                for col in cat_cols:
                    out[col]=out[col].astype('category').cat.codes

            # 6. Reindex columns so test columns match self.feature_columns exactly
            X=out.reindex(columns=self.feature_columns, fill_value=0)

            # 7. Scale using memorized self.scaler_mean and self.scaler_std
            if self.scale=='standard':
                for col in self.scaler_mean.keys():
                    if col in X.columns:
                        X[col]=(X[col] - self.scaler_mean[col]) / self.scaler_std[col]

            if has_target:
                return X, target_series
            return X
        except Exception as e:
            logger.error("Error during transform: %s", e)
            return df

    def fit_transform(self, df, **kwargs):
        self.fit(df, **kwargs)
        return self.transform(df)

    # ==========================
    # 3. Custom Models & Evaluation
    # ==========================
    def add_model(self, name, model):
        self.custom_models[name]=model
        logger.info("Registered custom model '%s'", name)
        return self

    def evaluate_baseline_models(self, X, y, cv=5):
        try:
            if not isinstance(y, pd.Series):
                y=pd.Series(y)

            is_classification=y.dtype=='object' or str(y.dtype).startswith('cat') or y.nunique() <= 20
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
                logger.info("Evaluating %d baseline models (task=%s, cv=%d)", len(models), task, cv)

                for name, model in models.items():
                    try:
                        acc=cross_val_score(model, X, y, cv=kf, scoring="accuracy")
                        f1=cross_val_score(model, X, y, cv=kf, scoring="f1_weighted")
                        results.append({"Model": name, "Accuracy": round(acc.mean(), 4), "F1": round(f1.mean(), 4)})
                    except Exception as e:
                        results.append({"Model": name, "Accuracy": None, "F1": str(e)})

                leaderboard=pd.DataFrame(results).sort_values(by="Accuracy", ascending=False)
            else:
                kf=KFold(n_splits=cv, shuffle=True, random_state=42)
                models={
                    "Ridge Regression": Ridge(),
                    "Decision Tree": DecisionTreeRegressor(random_state=42),
                    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
                    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
                }
                models.update(self.custom_models)
                logger.info("Evaluating %d baseline models (task=%s, cv=%d)", len(models), task, cv)

                for name, model in models.items():
                    try:
                        r2=cross_val_score(model, X, y, cv=kf, scoring="r2")
                        rmse=cross_val_score(model, X, y, cv=kf, scoring="neg_root_mean_squared_error")
                        results.append({"Model": name, "R2": round(r2.mean(), 4), "RMSE": round(-rmse.mean(), 4)})
                    except Exception as e:
                        results.append({"Model": name, "R2": None, "RMSE": str(e)})

                leaderboard=pd.DataFrame(results).sort_values(by="R2", ascending=False)

            logger.info("Baseline leaderboard:\n%s", leaderboard.to_string(index=False))
            return leaderboard
        except Exception as e:
            logger.error("Error in evaluate_baseline_models: %s", e)
