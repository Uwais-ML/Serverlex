import pandas as pd
import numpy as np
import logging 

logger=logging.getLogger("Clean")
if not logger.handlers:
    _h=logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(_h)
    logger.setLevel(logging.INFO)

class Clean:
    def __init__(self, path_or_df):
        if isinstance(path_or_df, pd.DataFrame):
            self.path = None
            self.data = path_or_df.copy()
        else:
            self.path = path_or_df
            self.data = None

    def load(self):
        try:
            if not hasattr(self, 'data') or self.data is None:
                if self.path is not None:
                    self.data=pd.read_csv(self.path)
        except Exception as e:
            print(e)
    def seperate(self):
        try:
            self.load()
            self.numerical_columns=self.data.select_dtypes(include='number')
            self.categrorical_columns=self.data.select_dtypes(include=['object', 'category', 'string'])
        except Exception as e:
            print(e)
    def outliers_IQR(self,param):
        self.seperate()
        Q1 = self.numerical_columns.quantile(0.25)
        Q3 = self.numerical_columns.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        condition=((self.numerical_columns >=lower_bound) & (self.numerical_columns<=upper_bound)).all(axis=1)
        if param=='r':
            self.data = self.data.loc[condition]
            self.numerical_columns = self.numerical_columns[condition]
            return
        if param=='m':
            column_mean=self.numerical_columns.mean()
            self.numerical_columns=self.numerical_columns.mask(~condition,column_mean,axis=1)
            self.data[self.numerical_columns.columns] = self.numerical_columns
            return
        column_median=self.numerical_columns.median()
        self.numerical_columns=self.numerical_columns.mask(~condition,column_median,axis=1)
        self.data[self.numerical_columns.columns]=self.numerical_columns
    def outliers_Z(self,param,threshold=None):
        self.seperate()
        mean=self.numerical_columns.mean()
        x_diff_mean=(self.numerical_columns)-mean
        std_dev=self.numerical_columns.std()
        z_score=x_diff_mean/std_dev
        t = threshold if threshold is not None else 3
        is_valid = z_score.abs() < t
        if param == 'r':
            condition = is_valid.all(axis=1)
            self.data = self.data.loc[condition]
            self.numerical_columns = self.numerical_columns.loc[condition]
            return
        elif param == 'm':
            self.numerical_columns = self.numerical_columns.mask(~is_valid, mean, axis=1)
            self.data[self.numerical_columns.columns] = self.numerical_columns
            return
        column_median=self.numerical_columns.median()
        self.numerical_columns=self.numerical_columns.mask(~is_valid,column_median,axis=1)
        self.data[self.numerical_columns.columns]=self.numerical_columns
    def Clean_string(self, data_time=None):
        try:
            self.seperate()

            # Handle date time format from user: data_time=["column name of date time", format you want it to be] or else None
            if data_time is not None:
                col_name = data_time[0]
                dt_format = data_time[1]
                if col_name in self.data.columns:
                    parsed_dates = pd.to_datetime(self.data[col_name], format='mixed', errors='coerce')
                    self.data[col_name] = parsed_dates.dt.strftime(dt_format)

            # Eliminate gaps between categories and lower them all for categorical columns
            cat_cols = [c for c in self.categrorical_columns.columns if data_time is None or c != data_time[0]]
            for col in cat_cols:
                self.data[col] = self.data[col].astype(str).str.lower().str.replace(r"\s+", "", regex=True)
                self.data[col] = self.data[col].mask(self.categrorical_columns[col].isna(), None)
            self.categrorical_columns = self.data[self.categrorical_columns.columns]
        except Exception as e:
            print(e)
    def delete_duplicates(self):
        try:
            self.load()
            self.data = self.data.drop_duplicates()
            self.seperate()
        except Exception as e:
            print(e)
    def fill_missing(self, param='m'):
        try:
            self.seperate()
            if param in ['r', 'd', 'drop']:
                self.data = self.data.dropna()
                self.seperate()
                return
            if param in ['m', 'mean']:
                fill_vals = self.numerical_columns.mean()
            else:
                fill_vals = self.numerical_columns.median()
            self.numerical_columns = self.numerical_columns.fillna(fill_vals)
            self.data[self.numerical_columns.columns] = self.numerical_columns
        except Exception as e:
            print(e)
    def fill_categorical_missing(self, fill_val=None):
        try:
            self.seperate()
            for col in self.categrorical_columns.columns:
                val = fill_val if fill_val is not None else (self.data[col].mode()[0] if not self.data[col].mode().empty else "unknown")
                self.data[col] = self.data[col].fillna(val)
            self.categrorical_columns = self.data[self.categrorical_columns.columns]
        except Exception as e:
            print(e)
    def clean_column_names(self):
        try:
            self.load()
            self.data.columns = (
                self.data.columns.str.strip()
                .str.lower()
                .str.replace(r"\s+", "_", regex=True)
                .str.replace(r"[^\w_]", "", regex=True)
            )
            self.seperate()
        except Exception as e:
            print(e)
    def drop_uninformative(self, null_threshold=0.7):
        try:
            self.load()
            null_pct = self.data.isnull().mean()
            cols_to_drop = null_pct[null_pct > null_threshold].index.tolist()
            zero_var = [col for col in self.data.columns if self.data[col].nunique(dropna=False) <= 1]
            self.data = self.data.drop(columns=list(set(cols_to_drop + zero_var)))
            self.seperate()
        except Exception as e:
            print(e)
    def clean_numeric_strings(self, columns):
        try:
            self.load()
            for col in columns:
                if col in self.data.columns:
                    cleaned = self.data[col].astype(str).str.replace(r"[\$,]", "", regex=True)
                    self.data[col] = pd.to_numeric(cleaned, errors="coerce")
            self.seperate()
        except Exception as e:
            print(e)
    def low_corr(self, target=None, threshold=0.1):
        try:
            self.seperate()
            corr_matrix = self.numerical_columns.corr().abs()
            if target is not None:
                if target in corr_matrix.columns:
                    low_corr_cols = corr_matrix[target][corr_matrix[target] < threshold].index.tolist()
                    self.data = self.data.drop(columns=low_corr_cols)
                    self.seperate()
            else:
                np.fill_diagonal(corr_matrix.values, 0)
                low_corr_cols = corr_matrix.columns[(corr_matrix.max() < threshold)].tolist()
                self.data = self.data.drop(columns=low_corr_cols)
                self.seperate()
        except Exception as e:
            print(e)
    def summary(self):
        try:
            self.load()
            missing_cols = (self.data.isnull().sum() > 0).sum()
            logger.info("Dataset summary: shape=%s, duplicates=%d, columns_with_missing=%d",
                        self.data.shape, self.data.duplicated().sum(), missing_cols)
        except Exception as e:
            logger.error("Error in summary: %s", e)
    def export(self, output_path):
        try:
            self.data.to_csv(output_path, index=False)
            logger.info("Saved cleaned dataset to '%s'", output_path)
        except Exception as e:
            logger.error("Error in export: %s", e)
    def total_clean(
        self,
        data_time=None,
        numeric_string_cols=None,
        missing_param='m',
        cat_fill_val=None,
        outlier_method='IQR',
        outlier_param='m',
        outlier_threshold=None,
        null_threshold=0.7,
        corr_target=None,
        corr_threshold=None,
        export_path=None
    ):
        try:
            # 1. Load data
            self.load()

            # 2. Standardize column names
            self.clean_column_names()

            # Normalize user-passed column names to match cleaned column names if needed
            def _normalize_col(c):
                import re
                return re.sub(r"[^\w_]", "", re.sub(r"\s+", "_", str(c).strip().lower()))

            if data_time is not None:
                c, fmt = data_time[0], data_time[1]
                if c not in self.data.columns and _normalize_col(c) in self.data.columns:
                    data_time = [_normalize_col(c), fmt]

            if numeric_string_cols:
                numeric_string_cols = [
                    _normalize_col(c) if c not in self.data.columns and _normalize_col(c) in self.data.columns else c
                    for c in numeric_string_cols
                ]

            if corr_target is not None:
                if corr_target not in self.data.columns and _normalize_col(corr_target) in self.data.columns:
                    corr_target = _normalize_col(corr_target)

            # 3. Remove duplicate rows
            self.delete_duplicates()

            # 4. Drop uninformative columns (high nulls or constant values)
            if null_threshold is not None:
                self.drop_uninformative(null_threshold=null_threshold)

            # 5. Clean currency / formatted numeric strings into actual numbers
            if numeric_string_cols:
                self.clean_numeric_strings(numeric_string_cols)

            # 6. Clean strings & format datetime
            self.Clean_string(data_time=data_time)

            # 7. Fill missing values (numerical)
            self.fill_missing(param=missing_param)

            # 8. Fill missing values (categorical)
            self.fill_categorical_missing(fill_val=cat_fill_val)

            # 9. Handle numerical outliers
            if outlier_method == 'Z':
                self.outliers_Z(param=outlier_param, threshold=outlier_threshold)
            elif outlier_method == 'IQR':
                self.outliers_IQR(param=outlier_param)

            # 10. Drop low correlation columns if requested
            if corr_threshold is not None:
                self.low_corr(target=corr_target, threshold=corr_threshold)

            # 11. Export if path specified
            if export_path:
                self.export(export_path)

            return self.data
        except Exception as e:
            print(e)
            return self.data

