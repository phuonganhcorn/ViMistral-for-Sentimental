from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from datasets import Dataset

def generate_prompt(data_point):
    return f"""
            [INST]
            Analyze the sentiment of the comment in enclosed in square brackets,
            determine if it is positive, neutral, or negative, and return the answer as
            the corresponding sentiment label "Positive" or "Neutral" or "Negative".[/INST]

            [{data_point["comment"]}] = {data_point["label"]}
            """.strip()

def generate_test_prompt(data_point):
    return f"""
            [INST]
            Analyze the sentiment of the comment in enclosed in square brackets,
            determine if it is positive, neutral, or negative, and return the answer as
            the corresponding sentiment label "Positive" or "Neutral" or "Negative".[/INST]

            [{data_point["comment"]}] = """.strip()



if __name__ == '__main__':
    train = pd.read_csv('final-train.csv')
    X_train = pd.DataFrame(train.apply(generate_prompt, axis=1), columns=["comment"])

    eval = pd.read_csv("final-dev.csv")
    X_eval = pd.DataFrame(eval.apply(generate_test_prompt, axis=1), columns=["comment"])

    test = pd.read_csv("final-test.csv")
    X_test = pd.DataFrame(test.apply(generate_test_prompt, axis=1), columns=["comment"])
    y_true = test['label']

    train_data = Dataset.from_pandas(X_train)
    eval_data = Dataset.from_pandas(X_eval)

    print(train_data)
    print(eval_data)