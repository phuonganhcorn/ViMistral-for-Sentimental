import matplotlib.pyplot as plt

# Create subplots with three rows and one column
fig, axes = plt.subplots(3, 1, figsize=(10, 15))

# Filter rows with 'n_stars' equal to 1 or 2
stars_1_2 = df[df['n_star'].isin([1, 2])]
axes[0].hist(stars_1_2['sentimental-score'], bins=20, color='red', alpha=0.7)
axes[0].set_xlabel('Sentiment Score')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Distribution of Sentiment Scores for 1 and 2-Star Ratings')
axes[0].grid(True)

# Filter rows with 'n_stars' equal to 3
stars_3 = df[df['n_star'] == 3]
axes[1].hist(stars_3['sentimental-score'], bins=20, color='green', alpha=0.7)
axes[1].set_xlabel('Sentiment Score')
axes[1].set_ylabel('Frequency')
axes[1].set_title('Distribution of Sentiment Scores for 3-Star Ratings')
axes[1].grid(True)

# Filter rows with 'n_stars' equal to 4 or 5
stars_4_5 = df[df['n_star'].isin([4, 5])]
axes[2].hist(stars_4_5['sentimental-score'], bins=20, color='blue', alpha=0.7)
axes[2].set_xlabel('Sentiment Score')
axes[2].set_ylabel('Frequency')
axes[2].set_title('Distribution of Sentiment Scores for 4 and 5 Star Ratings')
axes[2].grid(True)

# Adjust layout
plt.tight_layout()
plt.show()
