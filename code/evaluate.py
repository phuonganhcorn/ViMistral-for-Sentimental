def evaluate(y_true, y_pred):
    labels = ['Positive', 'Neutral', 'Negative']
    mapping = {'Positive': 2, 'Neutral': 1, 'Negative': 0}
    def map_func(x):
        return mapping.get(x, 1)

    y_true = np.vectorize(map_func)(y_true)
    y_pred = np.vectorize(map_func)(y_pred)

    # Calculate accuracy
    accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
    print(f'Accuracy: {accuracy:.3f}')

    # Generate accuracy report
    unique_labels = set(y_true)  # Get unique labels

    for label in unique_labels:
        label_indices = [i for i in range(len(y_true))
                        if y_true[i] == label]
        label_y_true = [y_true[i] for i in label_indices]
        label_y_pred = [y_pred[i] for i in label_indices]
        accuracy = accuracy_score(label_y_true, label_y_pred)
        print(f'Accuracy for label {label}: {accuracy:.3f}')

    # Generate classification report
    class_report = classification_report(y_true=y_true, y_pred=y_pred)
    print('\nClassification Report:')
    print(class_report)

    # Generate confusion matrix
    conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=[0, 1, 2])
    print('\nConfusion Matrix:')
    print(conf_matrix)

def predict(X_test, model, tokenizer):
    y_pred = []
    for i in tqdm(range(len(X_test))):
        prompt = X_test.iloc[i]["comment"]
        pipe = pipeline(task="text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens = 1,
                        temperature = 0.0,)
        result = pipe(prompt, pad_token_id=pipe.tokenizer.eos_token_id)
        answer = result[0]['generated_text'].split("=")[-1].lower()
        if "Positive" in answer:
            y_pred.append("Positive")
        elif "Negative" in answer:
            y_pred.append("Negative")
        elif "Neutral" in answer:
            y_pred.append("Neutral")
    return y_pred

# Little test and evaluate
y_pred = predict(X_test, model, tokenizer)
evaluate(y_true, y_pred)



y_pred = predict(X_test, model, tokenizer)
evaluate(y_true, y_pred)

evaluation = pd.DataFrame({'comment': X_test["comment"],
                        'y_true':y_true,
                        'y_pred': y_pred},
                        )
evaluation.to_csv("model-predict.csv", index=False)