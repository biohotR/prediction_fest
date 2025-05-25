import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import chi2_contingency


def detect_outliers_iqr(column, threshold=1.5):
    Q1 = column.quantile(0.25)
    Q3 = column.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    return column[(column < lower) | (column > upper)]


def is_continuous(column, unique_threshold=20):
    """
    Check if a column contains continuous data based on some factors.
    """

    if column.dtype in ['int64', 'float64']:
        if column.nunique() > unique_threshold:
            return True
        else:
            return False
    else:
        return False


def remove_outliers_iqr(df, columns, threshold=1.5):
    """
    Removes rows with outliers based on IQR from specified columns.
    """
    mask = pd.Series([True] * len(df))
    for col in columns:
        outliers = detect_outliers_iqr(df[col], threshold)
        mask &= ~df.index.isin(outliers.index)
    return df[mask]


def plot_all_boxplots(df, columns, title_prefix=""):
    for col in columns:
        plt.figure(figsize=(6, 4))
        sns.boxplot(y=df[col].dropna())
        plt.title(f"{title_prefix}{col}")
        plt.tight_layout()
        plt.show()


def melt_dataframe_for_boxplot(df, columns):
    melted = []
    for col in columns:
        values = df[col].dropna()
        for val in values:
            melted.append({'Feature': col, 'Value': val})
    return pd.DataFrame(melted)

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

def find_redundant_categorical_columns(cramer_matrix, threshold=0.9):
    """
    Identifies redundant pairs of categorical columns based on Cramér’s V values.
    Returns a list of columns suggested for removal.
    """
    redundant_cols = set()
    already_checked = set()
    
    for col1 in cramer_matrix.columns:
        for col2 in cramer_matrix.columns:
            if col1 == col2 or (col2, col1) in already_checked:
                continue
            already_checked.add((col1, col2))
            value = cramer_matrix.loc[col1, col2]
            if value >= threshold:
                print(f"🔁 Redundant pair: ({col1}, {col2}) → Cramér’s V = {value:.2f}")
                # Keep col1, suggest dropping col2
                redundant_cols.add(col2)
    
    return list(redundant_cols)

def find_redundant_categorical_columns_smart(cramer_matrix, threshold=0.9):
    """
    Detects redundant categorical columns using Cramér’s V,
    and prefers keeping the column with higher average correlation to others.
    """
    redundant_cols = set()
    already_checked = set()

    for col1 in cramer_matrix.columns:
        for col2 in cramer_matrix.columns:
            if col1 == col2 or (col2, col1) in already_checked:
                continue
            already_checked.add((col1, col2))
            value = cramer_matrix.loc[col1, col2]

            if value >= threshold:
                # Compute average correlation with other columns
                col1_avg_corr = cramer_matrix.loc[col1].drop([col1, col2]).astype(float).mean()
                col2_avg_corr = cramer_matrix.loc[col2].drop([col1, col2]).astype(float).mean()

                if col1_avg_corr >= col2_avg_corr:
                    to_drop = col2
                    to_keep = col1
                else:
                    to_drop = col1
                    to_keep = col2

                print(f"🔁 Redundant pair: ({col1}, {col2}) → V = {value:.2f}")
                print(f"   📊 Keeping: {to_keep}, ❌ Dropping: {to_drop} (less informative)")
                redundant_cols.add(to_drop)

    return list(redundant_cols)

def build_imputation_summary(df_original, df_final):
    """
    Builds a summary table with missing value stats and suggested imputation
    based on analysis from df_original and target columns in df_final.
    """
    total_rows = len(df_original)
    records = []

    for col in df_final.columns:
        if df_final[col].isna().sum() == 0:
            continue  # No missing values

        col_type = df_original[col].dtype
        is_numeric = col_type in [np.float64, np.int64]

        missing_count = df_original[col].isna().sum()
        missing_pct = (missing_count / total_rows) * 100

        outlier_count = count_outliers_iqr(df_original[col]) if is_numeric else None

        # Suggested method
        if not is_numeric:
            suggestion = "mode"
        elif missing_pct > 30:
            suggestion = "consider dropping"
        elif outlier_count is not None and outlier_count > total_rows * 0.05:
            suggestion = "median"
        else:
            suggestion = "mean"

        records.append({
            "Column": col,
            "Type": "numeric" if is_numeric else "categorical",
            "Missing Count": missing_count,
            "% Missing": round(missing_pct, 2),
            "Outliers": outlier_count,
            "Suggested Imputation": suggestion
        })

    return pd.DataFrame(records)

# Used for statistics and decision-making
def count_outliers_iqr(column, threshold=1.5):
    if column.dtype not in [np.float64, np.int64]:
        return 0
    col = column.dropna()
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - threshold * IQR
    upper = Q3 + threshold * IQR
    return int(((col < lower) | (col > upper)).sum())
