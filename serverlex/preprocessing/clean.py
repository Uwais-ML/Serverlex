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
        if not hasattr(self, 'data') or self.data is None:
            if self.path is not None:
                self.data = pd.read_csv(self.path)
            else:
                raise ValueError("No dataset or file path provided to Clean.")
        return self.data

    def separate(self):
        self.load()
        self.numerical_columns = self.data.select_dtypes(include='number')
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category', 'string'])
        self.categrorical_columns = self.categorical_columns  # Backward-compatibility typo alias
        return self.numerical_columns, self.categorical_columns

    seperate = separate  # Backward-compatibility typo alias

    def outliers_IQR(self, param='m', exclude=()):
        self.separate()
        exclude = set(exclude) if exclude else set()
        cols = [c for c in self.numerical_columns.columns if c not in exclude]
        if not cols:
            return self.data

        num = self.numerical_columns[cols]
        q1 = num.quantile(0.25)
        q3 = num.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        bad = (num < lower_bound) | (num > upper_bound)

        if param in ['r', 'drop', 'remove']:
            self.data = self.data.loc[~bad.any(axis=1)]
            self.separate()
            return self.data

        if param in ['m', 'mean']:
            fill = num.mean()
        else:
            fill = num.median()

        self.data[cols] = num.mask(bad, fill, axis=1)
        self.separate()
        return self.data

    def outliers_Z(self, param='m', threshold=None, exclude=()):
        self.separate()
        exclude = set(exclude) if exclude else set()
        cols = [c for c in self.numerical_columns.columns if c not in exclude]
        if not cols:
            return self.data

        num = self.numerical_columns[cols]
        mean = num.mean()
        std_dev = num.std()
        std_dev = std_dev.replace(0, 1.0)
        z_score = (num - mean) / std_dev
        t = threshold if threshold is not None else 3.0

        bad = z_score.abs() >= t

        if param in ['r', 'drop', 'remove']:
            self.data = self.data.loc[~bad.any(axis=1)]
            self.separate()
            return self.data

        if param in ['m', 'mean']:
            fill = mean
        else:
            fill = num.median()

        self.data[cols] = num.mask(bad, fill, axis=1)
        self.separate()
        return self.data

    def Clean_string(self, data_time=None):
        self.separate()

        # Handle date time format: data_time="col" or data_time=["col"] or data_time=["col", format]
        if data_time is not None:
            if isinstance(data_time, str):
                col_name, dt_format = data_time, None
            elif len(data_time) == 1:
                col_name, dt_format = data_time[0], None
            else:
                col_name, dt_format = data_time[0], data_time[1]

            if col_name in self.data.columns:
                parsed_dates = pd.to_datetime(self.data[col_name], format='mixed', errors='coerce')
                if dt_format:
                    self.data[col_name] = parsed_dates.dt.strftime(dt_format)
                else:
                    self.data[col_name] = parsed_dates

        # Standardize strings: strip leading/trailing, normalize internal whitespace to single space, lowercase
        cat_cols = [c for c in self.categorical_columns.columns if data_time is None or c != data_time[0]]
        for col in cat_cols:
            orig_na = self.data[col].isna()
            cleaned = self.data[col].astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)
            self.data[col] = cleaned.mask(orig_na, None)

        self.separate()
        return self.data

    def delete_duplicates(self):
        self.load()
        self.data = self.data.drop_duplicates()
        self.separate()
        return self.data
    def fill_missing(self, param='m'):
        self.separate()
        if param in ['r', 'd', 'drop']:
            self.data = self.data.dropna()
            self.separate()
            return self.data
        if param in ['m', 'mean']:
            fill_vals = self.numerical_columns.mean()
        else:
            fill_vals = self.numerical_columns.median()
        self.numerical_columns = self.numerical_columns.fillna(fill_vals)
        self.data[self.numerical_columns.columns] = self.numerical_columns
        self.separate()
        return self.data

    def fill_categorical_missing(self, fill_val=None):
        self.separate()
        for col in self.categorical_columns.columns:
            val = fill_val if fill_val is not None else (self.data[col].mode()[0] if not self.data[col].mode().empty else "unknown")
            self.data[col] = self.data[col].fillna(val)
        self.separate()
        return self.data

    def clean_column_names(self):
        self.load()
        self.data.columns = (
            self.data.columns.str.strip()
            .str.lower()
            .str.replace(r"\s+", "_", regex=True)
            .str.replace(r"[^\w_]", "", regex=True)
        )
        self.separate()
        return self.data

    def drop_uninformative(self, null_threshold=0.7):
        self.load()
        null_pct = self.data.isnull().mean()
        cols_to_drop = null_pct[null_pct > null_threshold].index.tolist()
        zero_var = [col for col in self.data.columns if self.data[col].nunique(dropna=False) <= 1]
        all_drop = list(set(cols_to_drop + zero_var))
        if all_drop:
            self.data = self.data.drop(columns=all_drop)
        self.separate()
        return self.data

    def clean_numeric_strings(self, columns):
        self.load()
        for col in columns:
            if col in self.data.columns:
                cleaned = self.data[col].astype(str).str.replace(r"[\$,]", "", regex=True).str.strip()
                self.data[col] = pd.to_numeric(cleaned, errors="coerce")
        self.separate()
        return self.data

    def low_corr(self, target=None, threshold=0.1):
        self.separate()
        corr_matrix = self.numerical_columns.corr().abs()
        if target is not None:
            if target in corr_matrix.columns:
                low_corr_cols = corr_matrix[target][corr_matrix[target] < threshold].index.tolist()
                self.data = self.data.drop(columns=low_corr_cols)
                self.separate()
        else:
            np.fill_diagonal(corr_matrix.values, 0)
            low_corr_cols = corr_matrix.columns[(corr_matrix.max() < threshold)].tolist()
            if low_corr_cols:
                self.data = self.data.drop(columns=low_corr_cols)
                self.separate()
        return self.data

    def summary(self):
        self.load()
        missing_cols = (self.data.isnull().sum() > 0).sum()
        logger.info("Dataset summary: shape=%s, duplicates=%d, columns_with_missing=%d",
                    self.data.shape, self.data.duplicated().sum(), missing_cols)
        return {
            "shape": self.data.shape,
            "duplicates": int(self.data.duplicated().sum()),
            "columns_with_missing": int(missing_cols)
        }

    def export(self, output_path):
        self.load()
        self.data.to_csv(output_path, index=False)
        logger.info("Saved cleaned dataset to '%s'", output_path)

    def total_clean(
        self,
        data_time=None,
        numeric_string_cols=None,
        missing_param='m',
        cat_fill_val=None,
        outlier_method='IQR',
        outlier_param='m',
        outlier_threshold=None,
        exclude_outliers=None,
        null_threshold=0.7,
        corr_target=None,
        corr_threshold=None,
        export_path=None
    ):
        # 1. Load data
        self.load()

        # 2. Standardize column names
        self.clean_column_names()

        def _normalize_col(c):
            import re
            return re.sub(r"[^\w_]", "", re.sub(r"\s+", "_", str(c).strip().lower()))

        if data_time is not None:
            if isinstance(data_time, str):
                c, fmt = data_time, None
            elif len(data_time) == 1:
                c, fmt = data_time[0], None
            else:
                c, fmt = data_time[0], data_time[1]

            if c not in self.data.columns and _normalize_col(c) in self.data.columns:
                data_time = [_normalize_col(c), fmt] if fmt else [_normalize_col(c)]

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

        # 9. Handle numerical outliers with target/exclude protection
        exclude_set = set()
        if exclude_outliers:
            exclude_set.update([_normalize_col(c) if _normalize_col(c) in self.data.columns else c for c in exclude_outliers])
        if corr_target:
            exclude_set.add(corr_target)

        if outlier_method and str(outlier_method).lower() != 'none':
            if outlier_method == 'Z':
                self.outliers_Z(param=outlier_param, threshold=outlier_threshold, exclude=exclude_set)
            elif outlier_method == 'IQR':
                self.outliers_IQR(param=outlier_param, exclude=exclude_set)

        # 10. Drop low correlation columns if requested
        if corr_threshold is not None:
            self.low_corr(target=corr_target, threshold=corr_threshold)

        # 11. Export if path specified
        if export_path:
            self.export(export_path)

        return self.data

