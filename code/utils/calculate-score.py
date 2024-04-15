import pandas as pd

def calculate_sentiment_score(row, aspect_columns):
    # Initialize variables
    score = 0
    count_none = 0

    # Define sentiment score mapping
    sentiment_mapping = {'Positive': 0.05, 'Neutral': 0, 'Negative': -0.05}
    naspect = len(aspect_columns)

    # Iterate over aspect columns to calculate sentiment score
    for aspect in aspect_columns:
        feedback = row[aspect]
        if feedback in sentiment_mapping:
            score += sentiment_mapping[feedback]

        if feedback == 'None':
            count_none += 1

        # The quality of feedback increases with its level of detail
        # Calculate the ratio of 'None' value and the number of aspects
        # Adjust the score based on the ratio of 'None' values
        if (count_none / naspect) <= 0.25:
            score += 0.05
        elif (count_none / naspect) >= 0.5:
            score -= 0.05

    # Adjust score based on star ratings
    if row['n_star'] in [1, 2]:
        score -= 0.25
    elif row['n_star'] in [4, 5]:
        score += 0.25

    return score

def normalize_score(sentiment_scores):
    # Define the ranges for each sentiment category
    positive_range = (0, 1)
    neutral_range = (-0.6, 0)
    negative_range = (-1, -0.4)
    max_score = max(sentiment_scores)


    # Normalize the scores based on the defined ranges
    normalized_scores = []
    for score in sentiment_scores:
        if score >= 0:
            normalized_score = positive_range[0] + (positive_range[1] - positive_range[0]) * (score / max_score)
        elif score >= -0.6:
            normalized_score = neutral_range[0] + (neutral_range[1] - neutral_range[0]) * ((score + 0.2) / 0.4)
        else:
            normalized_score = negative_range[0] + (negative_range[1] - negative_range[0]) * ((score + 0.2) / 0.4)
        normalized_scores.append(round(normalized_score, 2))

    return normalized_scores

if __name__ == '__main__':
    filename =  str(input('Input filename: '))
    # Load data
    data = pd.read_csv(filename)
    # Extract feature names from the DataFrame columns
    aspect_columns = [col for col in data.columns if col not in ['index', 'comment']]
    # Calculate sentiment scores for each comment
    sentiment_scores = []
    for _, row in data.iterrows():
        sentiment_scores.append(round(calculate_sentiment_score(row, aspect_columns), 2))

    # Add the normalized sentiment scores to the DataFrame
    data['sentimental-score'] = normalize_score(sentiment_scores)
    lower_filename = filename.lower()
    data.to_csv(f'final-{lower_filename}', index = False)
