import pandas as pd

def labeled_data(filename):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(filename)

    # Define ranges for sentiment scores
    positive_range = (0, 1)
    neutral_range = (-0.6, 0)
    negative_range = (-1, -0.4)

    # Initialize an empty list to store labeled data
    labeled_data = []

    # Iterate over each row in the DataFrame
    for _, row in data.iterrows():
        # Extract 'n_star' and 'sentimental-score' values from the row
        n_stars = row['n_star']
        score = row['sentimental-score']

        # Determine label based on star rating and sentiment score
        if n_stars in [4, 5]:
            if score >= positive_range[0] and score <= positive_range[1]:
                label = 'Positive'
            elif score >= negative_range[0] and score <= negative_range[1]:
                label = 'Neutral'
            elif score >= neutral_range[0] and score <= neutral_range[1]:
                label = 'Positive'
        elif n_stars == 3:
            if score >= positive_range[0] and score <= positive_range[1]:
                label = 'Positive'
            elif score >= neutral_range[0] and score <= neutral_range[1]:
                label = 'Neutral'
            elif score >= negative_range[0] and score <= negative_range[1]:
                label = 'Neutral'
        elif n_stars in [1, 2]:
            if score >= positive_range[0] and score <= positive_range[1]:
                label = 'Neutral'
            elif score >= neutral_range[0] and score <= neutral_range[1]:
                label = 'Negative'
            elif score >= negative_range[0] and score <= negative_range[1]:
                label = 'Negative'
        # Append label to labeled_data list
        labeled_data.append(label)

    # Return the labeled data
    return labeled_data

# Ask for filename input from the user
filename = str(input('Input filename: '))

print(labeled_data(filename))
# Call labeled_data function to label the data in the file
df = pd.read_csv(filename)
df['label'] = labeled_data(filename)  # Create new column 'label' with values from labeled_data

df.to_csv(filename, index = False)