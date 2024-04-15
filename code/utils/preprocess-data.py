import pandas as pd
import re

def remove_emojis(text):
    # Define a regular expression pattern to match emojis
    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F"  # emoticons
                        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                        u"\U0001F680-\U0001F6FF"  # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                        u"\U00002500-\U00002BEF"  # chinese char
                        u"\U00002702-\U000027B0"
                        u"\U00002702-\U000027B0"
                        u"\U000024C2-\U0001F251"
                        u"\U0001f926-\U0001f937"
                        u"\U00010000-\U0010ffff"
                        u"\u2640-\u2642"
                        u"\u2600-\u2B55"
                        u"\u200d"
                        u"\u23cf"
                        u"\u23e9"
                        u"\u231a"
                        u"\ufe0f"  # dingbats
                        u"\u3030"
                        "]+", flags=re.UNICODE)
    # Remove emojis using the pattern
    return emoji_pattern.sub(r'', text)

def preprocess_data(input_filename, output_filename):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_filename)

    # Remove emojis from the 'comment' column
    df['comment'] = df['comment'].apply(remove_emojis)

    # Extract columns
    rating_stars = df['n_star']
    labels = df['label']
    comments = df['comment']

    # Concatenate columns into a new DataFrame
    ndf = pd.concat([comments, labels, rating_stars], axis=1)

    # Extract features from labels and rename the column
    features = ndf['label'].str.extractall(r'(?P<Features>[A-Za-z&]+)#(?P<Feedback>Positive|Negative|Neutral)').reset_index(level=1, drop=True)

    # Pivot the DataFrame
    pivot_features = features.pivot_table(index=features.index, columns='Features', values='Feedback', aggfunc=lambda x: ' '.join(x)).fillna('None')

    # Merge the pivoted features DataFrame with the original DataFrame
    merged_df = pd.concat([ndf, pivot_features], axis=1)

    # Drop the 'label' column from the merged DataFrame
    merged_df = merged_df.drop('label', axis=1)

    # Drop rows with NaN values
    merged_df = merged_df.dropna()

    # Save the processed DataFrame to a CSV file
    merged_df.to_csv(output_filename, index=False)

    # Read the saved CSV file and display the DataFrame
    test = pd.read_csv(output_filename)
    print(test)

# Example usage
preprocess_data('Train.csv', 'train.csv')
preprocess_data('Test.csv', 'test.csv')
preprocess_data('Dev.csv', 'dev.csv')