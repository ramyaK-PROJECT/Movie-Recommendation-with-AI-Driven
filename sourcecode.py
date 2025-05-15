# Delivering Personalized Movie Recommendations with an AI-Driven Matchmaking System

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import gaussian_kde

# Load Dataset
df = pd.read_csv('movi.csv')

print (df.head())

# Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Ensure rating is numeric
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')

# Drop missing values
df_clean = df.dropna(subset=['rating'])

# ----------------------------
# 1. Line Plot: Average Rating Over Time
# ----------------------------
line_plot = df_clean.set_index('timestamp').resample('M')['rating'].mean()

plt.figure(figsize=(10, 6))
plt.plot(line_plot.index, line_plot.values, marker='o')
plt.title('Average Movie Rating Over Time')
plt.xlabel('Date')
plt.ylabel('Average Rating')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 2. Heatmap: Correlation Matrix
# ----------------------------
plt.figure(figsize=(8, 6))
corr = df_clean[['rating', 'age', 'release_year']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.show()

# ----------------------------
# 3. Matrix: User vs Movie Ratings
# ----------------------------
rating_matrix = df_clean.pivot_table(index='user_id', columns='movie_id', values='rating')
subset_matrix = rating_matrix.iloc[:10, :10]  # Show only sample for readability

plt.figure(figsize=(10, 8))
sns.heatmap(subset_matrix, cmap='YlGnBu', linewidths=.5)
plt.title('User-Movie Rating Matrix (Sample)')
plt.tight_layout()
plt.show()

# ----------------------------
# 4. KDE Diagram: Rating Distribution
# ----------------------------
kde_data_clean = df_clean['rating'].astype(float).dropna()
kde = gaussian_kde(kde_data_clean)
x_vals = np.linspace(kde_data_clean.min(), kde_data_clean.max(), 1000)
y_vals = kde(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, color='blue')
plt.fill_between(x_vals, y_vals, color='skyblue', alpha=0.5)
plt.title('Rating Distribution (KDE)')
plt.xlabel('Rating')
plt.ylabel('Density')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# 5. Box Plot: Ratings by Gender
# ----------------------------
plt.figure(figsize=(8, 6))
sns.boxplot(data=df_clean, x='gender', y='rating')
plt.title('Rating Distribution by Gender')
plt.tight_layout()
plt.show()