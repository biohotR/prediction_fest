import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from helpers import detect_outliers_iqr

# citire date in pandas DataFrame
df = pd.read_csv("data_files/pirvision_office_train.csv")


# Boxplot doar pentru coloanele cu valori extreme
df_extreme = df[['OBS_1']].melt(var_name='Feature', value_name='Value')

plt.figure(figsize=(6, 4))
sns.boxplot(data=df_extreme, x='Feature', y='Value')
plt.yscale('log')  # ✅ setăm scala logaritmică
plt.title("Boxplot pe scară logaritmică: OBS_1")
plt.tight_layout()
plt.show()
