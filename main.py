import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
from diptest import diptest
from scipy.stats import chi2_contingency
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.preprocessing import StandardScaler

from helpers import (build_imputation_summary, cramers_v, detect_outliers_iqr,
                     find_redundant_categorical_columns,
                     find_redundant_categorical_columns_smart, is_continuous,
                     melt_dataframe_for_boxplot, remove_outliers_iqr)

# from sklearn.metrics import classification_report


df = pd.read_csv("data_files/pirvision_office_train.csv")

numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns.to_list()
# categorical_columns = df.select_dtypes(include=['object', 'category']).columns.to_list()


### Analysis of the type of attributes and their values

# check which columns are continuous
continuous_columns = []
for col in numeric_columns:
    if is_continuous(df[col]) and col not in ['Class', 'Day Index']:
        continuous_columns.append(col)

# for the continuous columns, find out
# the number of examples from the set without missing values
# and the mean, standard deviation, min, max, 25th and 75th percentiles and the median

stats_continuous = {}
for col in continuous_columns:
    stats_continuous[col] = {
        'count': df[col].notna().sum(), # numarul de exemple din setul de date care nu au valori lipsa
        'mean': df[col].mean(),
        'std': df[col].std(),
        'min': df[col].min(),
        'max': df[col].max(),
        '25%': df[col].quantile(0.25),
        '75%': df[col].quantile(0.75),
        'median': df[col].median()
    }


# for the continuous columns, plot the distribution of the values

# Remove outliers first - this is a preprocessing step
df_clean = remove_outliers_iqr(df, continuous_columns)
df_long = melt_dataframe_for_boxplot(df_clean, continuous_columns)

plt.figure(figsize=(20, 6))
sns.boxplot(data=df_long, x='Feature', y='Value', showfliers=False)
plt.xticks(rotation=90)
plt.title("Boxplot combinat fără outlieri pentru toate coloanele continue")
plt.tight_layout()
plt.show()


### Discrete/Categorical Columns Analysis
# for discrete and categorical columns, extract and plot the number of 
# examples from the set without missing values
# anad the number of unique values


discrete_or_categorical_columns = ['Class', 'Day', 'Day Index', 'Temp (C)', 'Temp (F)']

# statistics
for col in discrete_or_categorical_columns:
    non_missing = df[col].notna().sum()
    unique_vals = df[col].nunique()
    print(f"{col}: {non_missing} valori non-lipsă, {unique_vals} valori unice")

    # graphs
    plt.figure(figsize=(6, 3))
    sns.countplot(x=col, data=df)
    plt.title(f"Distribuția valorilor pentru {col}")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




# count atributes with missing values
missing_values = df.isna().sum() # creeaza un tabel cu True/False pentru fiecare valoare si apoi face suma pe coloane


for col in df.columns:
    dtype = df[col].dtype
    nunique = df[col].nunique()
    # print(f"{col} → dtype: {dtype}, valori unice: {nunique}")



### Class Balance Analysis
df['Class'].value_counts().sort_index().plot.bar()
plt.title("Distribuția claselor în setul de antrenare")
plt.xlabel("Clasă")
plt.ylabel("Frecvență")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

# Calculate class frequencies and percentages
class_counts = df['Class'].value_counts()
print("Frecvența pe clase:")
print(class_counts)

class_ratios = class_counts / class_counts.sum()
print("\nProcentaj pe clase:")
print(class_ratios.round(4))

### Analysys of Correlation Between Features
## the goal is to find out if there are redundant features

## continuous columns

# correlation matrix
corr_matrix = df[continuous_columns].corr()

plt.figure(figsize=(12, 8))
plt.matshow(corr_matrix, fignum=1, cmap='coolwarm', vmin=-1, vmax=1)
plt.xticks(range(len(corr_matrix.columns)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix.columns)), corr_matrix.columns)
plt.colorbar()
plt.title("Matricea de corelație Pearson pentru atributele numerice", pad=20)
plt.tight_layout()
plt.show()


## discrete/categorical columns
# Chi-squared test for independence

for col1, col2 in itertools.combinations(discrete_or_categorical_columns, 2):
    contingency = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, expected = chi2_contingency(contingency)
    
    # print(f"\nChi-Square Test between {col1} and {col2}:")
    # print(f"  χ² = {chi2:.4f}, p-value = {p:.4f}")
    # if p < 0.05:
    #     print("  🔗 Sunt corelate (se respinge ipoteza de independență)")
    # else:
    #     print("  🚫 Nu sunt corelate (se acceptă ipoteza de independență)")

    plt.figure(figsize=(8, 6))
    sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Contingență între {col1} și {col2}")
    plt.xlabel(col2)
    plt.ylabel(col1)
    plt.tight_layout()
    plt.show()



cramer_matrix = pd.DataFrame(index=discrete_or_categorical_columns, columns=discrete_or_categorical_columns, dtype=float)

for col1, col2 in itertools.combinations(discrete_or_categorical_columns, 2):
    value = cramers_v(df[col1], df[col2])
    cramer_matrix.loc[col1, col2] = value
    cramer_matrix.loc[col2, col1] = value

np.fill_diagonal(cramer_matrix.values, 1.0)  # diagonala = 1

# Afișăm heatmap
plt.figure(figsize=(7, 6))
sns.heatmap(cramer_matrix.astype(float), annot=True, cmap="YlGnBu", vmin=0, vmax=1)
plt.title("Matricea Cramér’s V pentru atribute categorice")
plt.tight_layout()
plt.show()



### This is a preprocessing step to remove redundant categorical columns
# finding redundant categorical columns based on Cramér's V (discrete/categorical columns)
redundant_cols_discrete = find_redundant_categorical_columns_smart(cramer_matrix, threshold=0.8)
print("\n📌 Suggested columns to drop due to redundancy:")
print(redundant_cols_discrete)

# remove redundant columns
df_train = df_clean.drop(columns=redundant_cols_discrete)

imputation_table_helper = build_imputation_summary(df, df_train)

print("\n Imputation Summary:"
      "\n", imputation_table_helper)

mean_imputer = SimpleImputer(strategy="mean")
df_train[['OBS_1']] = mean_imputer.fit_transform(df_train[['OBS_1']])

# Pentru categorice (ex: Timestamp)
mode_imputer = SimpleImputer(strategy="most_frequent")
df_train[['Timestamp']] = mode_imputer.fit_transform(df_train[['Timestamp']])


numeric_cols = df_train.select_dtypes(include=['float64', 'int64']).columns
imp_iter = IterativeImputer(max_iter=10, random_state=0)

df_train[numeric_cols] = imp_iter.fit_transform(df_train[numeric_cols])


print("Any missing left in df_train?", df_train.isna().any().any())



### Data Standardization
features_to_scale = df_train.drop(columns=['Class']).select_dtypes(include=['float64', 'int64']).columns

scaler = StandardScaler()
df_train[features_to_scale] = scaler.fit_transform(df_train[features_to_scale])


### Removing redundant features for the numerical columns
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

high_corr_pairs = [(col, row) for col in upper_triangle.columns for row in upper_triangle.index
                   if abs(upper_triangle.loc[row, col]) > 0.9]

print("Corelații Pearson > 0.9:")
for col1, col2 in high_corr_pairs:
    print(f"{col1} ↔ {col2}: {corr_matrix.loc[col1, col2]:.2f}")

# Keep only every 3rd OBS feature
obs_cols = [f"OBS_{i}" for i in range(1, 58)]
cols_to_keep = [col for i, col in enumerate(obs_cols) if i % 3 == 0]
cols_to_drop = list(set(obs_cols) - set(cols_to_keep))

print("Dropping these correlated features:", cols_to_drop)

df_train = df_train.drop(columns=cols_to_drop)
