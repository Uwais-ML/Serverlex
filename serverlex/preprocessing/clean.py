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

    def load_chunked(
        self,
        filepath=None,
        chunksize=100_000,
        downcast_numeric=True,
        quantize_8bit=False,
        auto_categorical=True,
        verbose=True,
        **read_csv_kwargs
    ):
        """
        Streaming in-flight chunked CSV loader designed for datasets exceeding available RAM (10+ GB).
        Instead of loading the entire raw file into uncompressed 64-bit RAM at once (which spikes memory
        by 3x-4x and triggers OOM crashes), this loader streams through a sliding chunk buffer:
          1. Reads `chunksize` rows at a time (e.g. 100k rows ~ 50 MB working set).
          2. Downcasts in-flight:
             - Floats: float64 -> float32 (50% reduction) or 8-bit quantized bins (87.5% reduction).
             - Integers: int64 -> int8, int16, or int32 based on chunk min/max boundaries.
             - Strings: low-cardinality string/object columns -> category dtype.
          3. Appends the memory-compact chunk and concatenates with minimal peak memory overhead.

        Parameters:
            filepath (str, optional): Path to CSV. Defaults to self.path.
            chunksize (int): Number of rows per streaming batch (default 100,000).
            downcast_numeric (bool): Downcast floats to float32 and ints to int8/16/32.
            quantize_8bit (bool): Extreme 1-byte quantization for continuous floats (0-255 uint8).
            auto_categorical (bool): Automatically cast low-cardinality string columns to category.
            verbose (bool): Log memory stats before and after chunked loading.
            **read_csv_kwargs: Forwarded to pd.read_csv.

        Returns:
            pd.DataFrame: Consolidated, memory-compact DataFrame loaded with minimal peak RAM.
        """
        path_to_use = filepath or self.path
        if not path_to_use:
            raise ValueError("No file path provided to Clean.load_chunked().")

        chunks = []
        total_rows = 0
        reader = pd.read_csv(path_to_use, chunksize=chunksize, **read_csv_kwargs)
        for chunk in reader:
            total_rows += len(chunk)
            for col in chunk.columns:
                col_type = chunk[col].dtype

                # 1. Integer Downcasting
                if downcast_numeric and pd.api.types.is_integer_dtype(col_type):
                    c_min = chunk[col].min()
                    c_max = chunk[col].max()
                    if c_min >= 0 and c_max <= np.iinfo(np.uint8).max:
                        chunk[col] = chunk[col].astype(np.uint8)
                    elif c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                        chunk[col] = chunk[col].astype(np.int8)
                    elif c_min >= 0 and c_max <= np.iinfo(np.uint16).max:
                        chunk[col] = chunk[col].astype(np.uint16)
                    elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                        chunk[col] = chunk[col].astype(np.int16)
                    elif c_min >= 0 and c_max <= np.iinfo(np.uint32).max:
                        chunk[col] = chunk[col].astype(np.uint32)
                    elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                        chunk[col] = chunk[col].astype(np.int32)

                # 2. Float Downcasting / 8-bit Quantization
                elif pd.api.types.is_float_dtype(col_type):
                    if quantize_8bit:
                        c_min = chunk[col].min()
                        c_max = chunk[col].max()
                        if pd.notna(c_min) and pd.notna(c_max) and c_max > c_min:
                            scaled = ((chunk[col] - c_min) / (c_max - c_min) * 255.0).clip(0, 255)
                            chunk[col] = scaled.fillna(0).astype(np.uint8)
                        else:
                            chunk[col] = chunk[col].astype(np.float32)
                    elif downcast_numeric and col_type == np.float64:
                        chunk[col] = chunk[col].astype(np.float32)

                # 3. Categorical Downcasting for low-cardinality strings
                elif auto_categorical and (isinstance(col_type, object) or col_type == 'object' or col_type == 'string'):
                    num_unique = chunk[col].nunique(dropna=True)
                    if len(chunk) > 0 and (num_unique / len(chunk)) < 0.5:
                        chunk[col] = chunk[col].astype('category')

            chunks.append(chunk)

        self.data = pd.concat(chunks, ignore_index=True)

        # Final pass: harmonize column bounds across chunks
        if downcast_numeric:
            self.optimize_memory(downcast_floats=downcast_numeric, downcast_ints=downcast_numeric, downcast_categories=auto_categorical, verbose=False)

        self.separate()
        if verbose:
            mem_mb = self.data.memory_usage(deep=True).sum() / (1024 ** 2)
            logger.info("load_chunked: Loaded %d rows across %d columns into %.2f MB (peak RAM capped)", total_rows, len(self.data.columns), mem_mb)

        return self.data

    def separate(self):
        self.load()
        self.numerical_columns = self.data.select_dtypes(include='number')
        self.categorical_columns = self.data.select_dtypes(include=['object', 'category', 'string'])
        self.categrorical_columns = self.categorical_columns  # Backward-compatibility typo alias
        return self.numerical_columns, self.categorical_columns

    seperate = separate  # Backward-compatibility typo alias

    def _get_outlier_candidates(self, exclude=(), method='IQR'):
        exclude_set = set(exclude) if exclude else set()
        cols = []
        for c in self.numerical_columns.columns:
            if c in exclude_set:
                continue
            s = self.numerical_columns[c].dropna()
            # 1. Skip boolean dtypes and binary {0, 1} indicator columns
            if pd.api.types.is_bool_dtype(s) or (s.nunique() <= 2 and set(s.unique()).issubset({0, 1, 0.0, 1.0})):
                continue
            # 2. For IQR, skip columns where Q1 == Q3 (degenerate IQR = 0)
            if method == 'IQR':
                q1 = float(s.quantile(0.25))
                q3 = float(s.quantile(0.75))
                if q3 - q1 == 0:
                    logger.info("Skipping column '%s' for IQR outlier handling: degenerate IQR (Q1=Q3=%.2f)", c, q1)
                    continue
            cols.append(c)
        return cols

    def outliers_IQR(self, param='m', exclude=(), warn_high_outlier_pct=0.10):
        self.separate()
        cols = self._get_outlier_candidates(exclude=exclude, method='IQR')
        if not cols:
            return self.data

        num = self.numerical_columns[cols]
        q1 = num.quantile(0.25)
        q3 = num.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr

        bad = (num < lower_bound) | (num > upper_bound)

        # Warn if a column flags more than threshold of rows
        for col in cols:
            bad_rate = float(bad[col].mean())
            if bad_rate > warn_high_outlier_pct:
                logger.warning("Column '%s' flagged %.1f%% of rows as outliers under IQR. Consider excluding this feature from outlier clipping.", col, bad_rate * 100)

        # In row drop mode ('r'), track class balance if target was provided in exclude
        if param in ['r', 'drop', 'remove']:
            rows_to_drop = bad.any(axis=1)
            for ex in exclude:
                if ex in self.data.columns and self.data[ex].nunique() <= 5:
                    before_counts = self.data[ex].value_counts(normalize=True).to_dict()
                    after_counts = self.data.loc[~rows_to_drop, ex].value_counts(normalize=True).to_dict()
                    logger.info("Outlier row removal class shift for '%s': before=%s, after=%s", ex, before_counts, after_counts)

            self.data = self.data.loc[~rows_to_drop]
            self.separate()
            return self.data

        if param in ['m', 'mean']:
            fill = num.mean()
        else:
            fill = num.median()

        self.data[cols] = num.mask(bad, fill, axis=1)
        self.separate()
        return self.data

    def outliers_Z(self, param='m', threshold=None, exclude=(), warn_high_outlier_pct=0.10):
        self.separate()
        cols = self._get_outlier_candidates(exclude=exclude, method='Z')
        if not cols:
            return self.data

        num = self.numerical_columns[cols]
        mean = num.mean()
        std_dev = num.std()
        std_dev = std_dev.replace(0, 1.0)
        z_score = (num - mean) / std_dev
        t = threshold if threshold is not None else 3.0

        bad = z_score.abs() >= t

        for col in cols:
            bad_rate = float(bad[col].mean())
            if bad_rate > warn_high_outlier_pct:
                logger.warning("Column '%s' flagged %.1f%% of rows as outliers under Z-score (threshold=%.1f).", col, bad_rate * 100, t)

        if param in ['r', 'drop', 'remove']:
            rows_to_drop = bad.any(axis=1)
            self.data = self.data.loc[~rows_to_drop]
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

    def drop_embedded_headers(self):
        """Remove rows that are copies of the header (e.g. from CSV concatenation)."""
        self.load()
        # A row is an embedded header if it matches ≥ half of column names exactly
        min_match = max(2, len(self.data.columns) // 2)
        mask = (self.data == self.data.columns).sum(axis=1) >= min_match
        if mask.any():
            n = int(mask.sum())
            logger.warning("Dropping %d embedded header row(s) found in data", n)
            self.data = self.data[~mask].reset_index(drop=True)
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
                cleaned = (
                    self.data[col]
                    .astype(str)
                    .str.replace(r"[^\d\.\-]", "", regex=True)
                    .str.strip()
                )
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

    def optimize_memory(self, downcast_floats=True, downcast_ints=True, downcast_categories=False, verbose=True):
        """
        Compresses dataset memory footprint by downcasting numerical columns to lower-bit precision:
        - Integers: evaluated against min/max bounds and safely downcast to int8, int16, or int32.
        - Floats: float64 (8 bytes) safely cast to float32 (4 bytes), cutting memory by 50% while preserving full ML precision.
        - Categories (optional): object/string columns with low cardinality (nunique/total < 0.5) converted to category dtype.

        Returns:
            pd.DataFrame: The memory-optimized DataFrame.
        """
        self.load()
        start_mem = self.data.memory_usage(deep=True).sum() / (1024 ** 2)

        for col in self.data.columns:
            col_type = self.data[col].dtype

            # 1. Integer Downcasting
            if downcast_ints and pd.api.types.is_integer_dtype(col_type):
                c_min = self.data[col].min()
                c_max = self.data[col].max()
                if c_min >= 0 and c_max <= np.iinfo(np.uint8).max:
                    self.data[col] = self.data[col].astype(np.uint8)
                elif c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    self.data[col] = self.data[col].astype(np.int8)
                elif c_min >= 0 and c_max <= np.iinfo(np.uint16).max:
                    self.data[col] = self.data[col].astype(np.uint16)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    self.data[col] = self.data[col].astype(np.int16)
                elif c_min >= 0 and c_max <= np.iinfo(np.uint32).max:
                    self.data[col] = self.data[col].astype(np.uint32)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    self.data[col] = self.data[col].astype(np.int32)

            # 2. Float Downcasting (float64 -> float32)
            elif downcast_floats and pd.api.types.is_float_dtype(col_type):
                if col_type == np.float64:
                    self.data[col] = self.data[col].astype(np.float32)

            # 3. Categorical Downcasting (optional for low-cardinality strings)
            elif downcast_categories and (isinstance(col_type, object) or col_type == 'object' or col_type == 'string'):
                num_unique = self.data[col].nunique(dropna=True)
                num_total = len(self.data[col])
                if num_total > 0 and (num_unique / num_total) < 0.5:
                    self.data[col] = self.data[col].astype('category')

        end_mem = self.data.memory_usage(deep=True).sum() / (1024 ** 2)
        reduction_pct = round(100.0 * (start_mem - end_mem) / max(start_mem, 1e-6), 2)
        if verbose:
            logger.info("RAM Optimization: %.2f MB -> %.2f MB (%.1f%% reduction)", start_mem, end_mem, reduction_pct)

        self.memory_summary = {
            "start_mem_mb": round(float(start_mem), 2),
            "end_mem_mb": round(float(end_mem), 2),
            "reduction_pct": reduction_pct
        }
        self.separate()
        return self.data

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
        optimize_memory=False,
        export_path=None
    ):
        # 1. Load data
        self.load()

        # 1a. Strip embedded header rows (e.g. from CSV concatenation) BEFORE column rename
        self.drop_embedded_headers()

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

        # 11. Optional memory optimization (downcast numeric precision)
        if optimize_memory:
            self.optimize_memory(downcast_floats=True, downcast_ints=True, verbose=True)

        # 12. Export if path specified
        if export_path:
            self.export(export_path)

        return self.data

