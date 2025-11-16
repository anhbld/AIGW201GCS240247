import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from io import StringIO

data = """Country,Age,Salary,Purchased
France,44,72000,No
Spain,27,48000,Yes
Germany,30,54000,No
Spain,38,61000,No
Germany,40,,Yes
France,35,58000,Yes
Spain,,52000,No
France,48,79000,Yes
Germany,50,83000,No
France,37,67000,Yes"""

df = pd.read_csv(StringIO(data))

df['Age']      = pd.to_numeric(df['Age'], errors='coerce')
df['Salary']   = pd.to_numeric(df['Salary'], errors='coerce')
df['Purchased']= df['Purchased'].map({'Yes':1, 'No':0})
df_clean = df.dropna(subset=['Salary'])
sns.set_style("whitegrid")
fig, ax = plt.subplots(2, 2, figsize=(12,10))

avg_salary = df_clean.groupby('Country')['Salary'].mean().reset_index()
sns.barplot(x='Country', y='Salary', data=avg_salary, ax=ax[0,0])
ax[0,0].set_title('Average Salary by Country')

sns.boxplot(x='Country', y='Salary', data=df_clean, ax=ax[0,1])
ax[0,1].set_title('Salary Distribution by Country')

sns.scatterplot(x='Age', y='Salary', hue='Purchased',
                palette={1:'green', 0:'red'}, s=100, data=df_clean, ax=ax[1,0])
ax[1,0].set_title('Age vs Salary (Purchase = Green)')

purchase_rate = df_clean.groupby('Country')['Purchased'].mean().reset_index()
sns.barplot(x='Country', y='Purchased', data=purchase_rate,
            palette='viridis', ax=ax[1,1])
ax[1,1].set_title('Purchase Rate by Country')
ax[1,1].set_ylabel('Rate (0‑1)')

plt.tight_layout()
plt.show()