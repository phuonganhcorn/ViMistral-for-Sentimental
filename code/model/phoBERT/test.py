from code.model.phoBERT.SentimentalClassifier import SentimentClassifier
import torch
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from torch.utils.data import Dataset, DataLoader
from code.model.phoBERT.SentimentalDataset import SentimentDataset

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging
def test(data_loader):
    skf = StratifiedKFold(n_splits=3)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    models = []
    for fold in range(skf.n_splits):
        model = SentimentClassifier(n_classes=7)
        model.to(device)
        model.load_state_dict(torch.load(f'phobert_fold{fold+1}.pth'))
        model.eval()
        models.append(model)

    texts = []
    predicts = []
    predict_probs = []
    real_values = []

    for data in data_loader:
        text = data['text']
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_masks'].to(device)
        targets = data['targets'].to(device)

        total_outs = []
        for model in models:
            with torch.no_grad():
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                total_outs.append(outputs)
        
        total_outs = torch.stack(total_outs)
        _, pred = torch.max(total_outs.mean(0), dim=1)
        texts.extend(text)
        predicts.extend(pred)
        predict_probs.extend(total_outs.mean(0))
        real_values.extend(targets)
    
    predicts = torch.stack(predicts).cpu()
    predict_probs = torch.stack(predict_probs).cpu()
    real_values = torch.stack(real_values).cpu()
    print(classification_report(real_values, predicts))
    return real_values, predicts

def check_wrong(real_values, predicts):
    wrong_arr = []
    wrong_label = []
    for i in range(len(predicts)):
        if predicts[i] != real_values[i]:
            wrong_arr.append(i)
            wrong_label.append(predicts[i])
    return wrong_arr, wrong_label




if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    test_dataset = SentimentDataset(test_df, tokenizer, max_len=50)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, num_workers=2)
    real_values, predicts = test(test_loader)
    
    class_names = ['Positive', 'Neutral', 'Negative']
    for i in range(15):
        print('-'*50)
        wrong_arr, wrong_label = check_wrong(real_values, predicts)
        print(test_df.iloc[wrong_arr[i]].comment)
        print(f'Predicted: ({class_names[wrong_label[i]]}) --vs-- Real label: ({class_names[real_values[wrong_arr[i]]]})')
