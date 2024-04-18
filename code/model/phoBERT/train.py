import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from code.model.phoBERT.SentimentalDataset import SentimentDataset
from code.model.phoBERT.SentimentalClassifier import SentimentClassifier
from gensim.utils import simple_preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix

import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader

from transformers import get_linear_schedule_with_warmup, AutoTokenizer, AutoModel, logging

import warnings
warnings.filterwarnings("ignore")

logging.set_verbosity_error()

def seed_everything(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

seed_everything(86)




def get_data(path):
    df = pd.read_csv(path)
    df = df[['label', 'comment']]  # Only extract 'label' and 'comment' columns
    df.columns = ['emotion', 'comment']  # Rename 'label' column to 'emotion'
    return df



def train(model, criterion, optimizer, train_loader):
    model.train()
    losses = []
    correct = 0

    for data in train_loader:
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_masks'].to(device)
        targets = data['targets'].to(device)

        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(outputs, targets)
        _, pred = torch.max(outputs, dim=1)

        correct += torch.sum(pred == targets)
        losses.append(loss.item())
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

    print(f'Train Accuracy: {correct.double()/len(train_loader.dataset)} Loss: {np.mean(losses)}')

def eval(test_data = False):
    model.eval()
    losses = []
    correct = 0

    with torch.no_grad():
        data_loader = test_loader if test_data else valid_loader
        for data in data_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_masks'].to(device)
            targets = data['targets'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, pred = torch.max(outputs, dim=1)

            loss = criterion(outputs, targets)
            correct += torch.sum(pred == targets)
            losses.append(loss.item())
    
    if test_data:
        print(f'Test Accuracy: {correct.double()/len(test_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(test_loader.dataset)
    else:
        print(f'Valid Accuracy: {correct.double()/len(valid_loader.dataset)} Loss: {np.mean(losses)}')
        return correct.double()/len(valid_loader.dataset)
    
def prepare_loaders(df, fold):
    tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base", use_fast=False)
    df_train = df[df.kfold != fold].reset_index(drop=True)
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    train_dataset = SentimentDataset(df_train, tokenizer, max_len=120)
    valid_dataset = SentimentDataset(df_valid, tokenizer, max_len=120)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=True, num_workers=2)
    
    return train_loader, valid_loader
    
if __name__ == '__main__':
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    EPOCHS = 6
    N_SPLITS = 3
    
    train_df = get_data('./normalized-train.csv')
    valid_df = get_data('./normalized-dev.csv')
    test_df = get_data('./normalized-test.csv')

    # We will use Kfold later
    train_df = pd.concat([train_df, valid_df], ignore_index=True)
    skf = StratifiedKFold(n_splits=N_SPLITS)
    for fold, (_, val_) in enumerate(skf.split(X=train_df, y=train_df.emotion)):
        train_df.loc[val_, "kfold"] = fold
        
    for fold in range(skf.n_splits):
        print(f'-----------Fold: {fold+1} ------------------')
        train_loader, valid_loader = prepare_loaders(train_df, fold=fold)
        model = SentimentClassifier(n_classes=3).to(device)
        criterion = nn.CrossEntropyLoss()
        # Recommendation by BERT: lr: 5e-5, 2e-5, 3e-5
        # Batchsize: 16, 32
        optimizer = AdamW(model.parameters(), lr=2e-5)
        
        lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=0, 
                    num_training_steps=len(train_loader)*EPOCHS
                )
        best_acc = 0
        for epoch in range(EPOCHS):
            print(f'Epoch {epoch+1}/{EPOCHS}')
            print('-'*30)

            train(model, criterion, optimizer, train_loader)
            val_acc = eval()

            if val_acc > best_acc:
                torch.save(model.state_dict(), f'phobert_fold{fold+1}.pth')
                # Copy additional files


                best_acc = val_acc
                