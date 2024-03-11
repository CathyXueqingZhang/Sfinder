import ast
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader,random_split,Dataset,TensorDataset
from torch.optim import Optimizer
from clip import CLIPModel
from attention import *
import numpy as np
import matplotlib.pyplot as plt
from model_2 import *
import pickle
import argparse


import os
import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

temperature = 0.3

#n_head, d_model, d_k, d_v = 4, 1536, 1024, 1024
n_head = 16  # Number of attention heads
d_model = 1536  # The embedding dimension
d_k = d_v = dim_head = d_model // n_head  # Dimension of key/query/value per head
d_output = 256

batch_size = 32



#github_diff = pd.read_csv("../output/finaldata/emb/code_diff_chunk_all.csv")
github_diff_train = pd.read_csv("../output/finaldata/dataset/train_diff.csv")
#github_message = pd.read_csv("../output/finaldata/emb/message_f.csv")
cve_df = pd.read_csv('../output/finaldata/dataset/cve_side_all.csv')
#cve_test = pd.read_csv("../output/finaldata/emb/cve_sid_em_test.csv")
file = open("../output/finaldata/final_output.txt","a+")
print("finish loaded")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def batch_generator(data, batch_size=2000):
    total_size = len(data)
    for start in range(0, total_size, batch_size):
        end = min(start + batch_size, total_size)
        yield data.iloc[start:end]

#get dataset matching
class CVEGitHubDataset(Dataset):
    def __init__(self, cve_df, github_df):
        self.github_data = github_df
        # Merge the datasets on 'CVE_ID', duplicating 'cve_df' entries as needed
        self.combined_data = pd.merge(self.github_data, cve_df, on='CVE_ID', how='inner')


    def _preprocess_jsonl(self, jsonl_file_path):
            data = []
            with open(jsonl_file_path, 'r') as file:
                for line in file:
                    record = json.loads(line)
                    # Deserialize the embedding if it's a string
                    if isinstance(record['embedding'], str):
                        record['embedding'] = ast.literal_eval(record['embedding'])
                    if isinstance(record['embedding_message'], str):
                        record['embedding_message'] = ast.literal_eval(record['embedding_message'])
                    # Add the record only if 'label' is present and equals 1
                    data.append(record)
            return pd.DataFrame(data)

    def __len__(self):
        return self.combined_data['CVE_ID'].nunique()

    def __getitem__(self, idx):
        row = self.combined_data.iloc[idx]

        cve_embedding = ast.literal_eval(row['cve_embedding'])
        cve_embedding_f = torch.tensor(cve_embedding, dtype=torch.float).view(1, -1)

        message_embedding = ast.literal_eval(row['message_embedding'])
        message_embedding_f = torch.tensor(message_embedding, dtype=torch.float).view(1, -1)


        diff_embedding = ast.literal_eval(row['embedding'])
        diff_embedding_f = torch.tensor(diff_embedding, dtype=torch.float)

        return cve_embedding_f, message_embedding_f, diff_embedding_f



class AvgMeter:
    "provided is designed to track and compute the average of a metric over time"
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]
def train_epoch(model, train_loader, optimizer, lr_scheduler, step, device):
    model.to(device)
    model.train()
    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()
    tqdm_object = tqdm(train_loader, total=len(train_loader))

    for cve_embeddings, github_message, github_diff in tqdm_object:
        cve_embeddings = cve_embeddings.to(device)  # Shape: (batch_size, 1, embedding_size)
        github_message = github_message.to(device)  # Shape: (batch_size, seq_len, embedding_size)
        github_diff = github_diff.to(device)  # Shape: (batch_size, seq_len, embedding_size)

        optimizer.zero_grad()

        loss,pooled_cve_embeddings,pooled_github_embeddings = model(cve_embeddings, github_message, github_diff)


        loss.backward()
        optimizer.step()

        count = cve_embeddings.size(0)

        accuracy = model.predict(pooled_cve_embeddings, pooled_github_embeddings,top_n=1)
        accuracy_meter.update(accuracy, count)

        '''if step == "batch":
            lr_scheduler.step()'''

        # Calculate accuracy
        '''with torch.no_grad():
            github_message_list = github_message.split(1, dim=0)
            github_diff_list = github_diff.split(1, dim=0)
            github_data = list(zip(github_message_list, github_diff_list))
            accuracy = model.predict(cve_embeddings, github_message,github_diff)
            accuracy_meter.update(accuracy, cve_embeddings.size(0))'''

        loss_meter.update(loss.item(), count)

        tqdm_object.set_postfix(train_loss=loss_meter.avg, lr=get_lr(optimizer))

    return loss_meter,accuracy_meter


def validate_epoch(model, val_loader, device):
    model.to(device)
    model.eval()

    loss_meter = AvgMeter()
    accuracy_meter = AvgMeter()

    with torch.no_grad():
        for cve_embeddings, github_message, github_diff in val_loader:
            cve_embeddings = cve_embeddings.to(device)  # Shape: (batch_size, 1, embedding_size)
            github_message = github_message.to(device)  # Shape: (batch_size, seq_len, embedding_size)
            github_diff = github_diff.to(device)  # Shape: (batch_size, seq_len, embedding_size)

            # Forward pass
            loss,pooled_cve_embeddings,pooled_github_embeddings = model(cve_embeddings, github_message, github_diff)

            # Calculate accuracy
            '''predicted_indices = model.predict(cve_embeddings, github_message, github_diff)
            correct_predictions = (predicted_indices == torch.arange(cve_embeddings.size(0)).to(device)).sum().item()
            accuracy = correct_predictions / cve_embeddings.size(0)'''
            accuracy = model.predict(pooled_cve_embeddings, pooled_github_embeddings)
            accuracy_meter.update(accuracy, cve_embeddings.size(0))

            count = cve_embeddings.size(0)
            loss_meter.update(loss.item(), count)
    return loss_meter, accuracy_meter






if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='input argument')
    parser.add_argument('--learning_rate', type=float, default=0.001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--i', type=int, default=0, metavar='N',
                        help='i number')
    args = parser.parse_args()

    learning_rate = args.learning_rate
    i = args.i

    model_path = '../best_model_'+ str(i) +'.pth'
    dataset = CVEGitHubDataset(cve_df, github_diff_train)
    print("finish dataset")

    # Split the dataset into training and validation sets
    train_size = int(0.8 * len(dataset))
    #val_size = int(0.1 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    #initialize
    model = CombinedModel(n_head, d_model, d_k, d_v, d_output, temperature)

    model.to(device)

    '''optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.001
    )'''
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=3, factor=0.3
    )

    #train
    num_epochs = 100
    best_val_accuracy = float('inf')

    train_losses, val_losses = [], []
    train_accuracies,val_accuracies = [], []

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        # Training Phase
        model.train()
        train_loss,train_accuracy = train_epoch(model, train_loader, optimizer, lr_scheduler, 'epoch', device)
        train_losses.append(train_loss.avg)
        train_accuracies.append(train_accuracy.avg)


        # Validation Phase
        model.eval()
        val_loss, val_accuracy = validate_epoch(model, val_loader, device)

        val_losses.append(val_loss.avg)
        val_accuracies.append(val_accuracy.avg)

        print(f"Training Loss: {train_loss.avg:.8f}, Accuracy: {train_accuracy.avg:.8f}")
        #print(f"Training Loss: {train_loss.avg:.4f}")
        print(f"Validation Loss: {val_loss.avg:.8f}, Accuracy: {val_accuracy.avg:.8f}")

        # Learning Rate Scheduler Step
        lr_scheduler.step(val_loss.avg)

        # Save the best model
        #if val_accuracy.avg > best_val_accuracy:
            #best_val_accuracy = val_accuracy.avg
        torch.save(model.state_dict(), model_path)

        print(f"Current LR: {get_lr(optimizer)}")

    #plot
    # Plotting training and validation losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.savefig(f'training_validation_loss_{i}.png', dpi=300)

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label='Train Accuarcy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy Over Epochs')
    plt.legend()
    plt.savefig(f'training_validation_acc_{i}.png', dpi=300)
'''



    # predict
    # Load dataset
    cve_t= pd.read_csv("../output/finaldata/emb/cve_sid_em_test.csv")
    test_cve = cve_t[['CVE_ID', 'embedding']]
    test_cve['embedding'] = test_cve['embedding'].apply(ast.literal_eval)
    test_cve_tuples_list = [row for row in test_cve.itertuples(index=False, name=None)]
    test_github_path = "../output/finaldata/emb/github_side_test.jsonl"


    def convert_jsonl_to_tuples(file_path):
        tuples_list = []
        with open(file_path, 'r') as file:
            for line in file:
                data = json.loads(line)
                cve_id = data.get("CVE_ID")
                diff_embedding = ast.literal_eval(data.get("diff_embedding"))
                message_embedding = ast.literal_eval(data.get("embedding"))
                tuples_list.append((cve_id, diff_embedding, message_embedding))
        return tuples_list


    # Apply this function to the embedding and diff_embedding columns
    github_embeddings = convert_jsonl_to_tuples(test_github_path)

    model = CombinedModel(n_head, d_model, d_k, d_v, dim_head, d_output, temperature)
    model.load_state_dict(torch.load('../best_model_1.pth'))
    model.to(device)

    github_embeding = get_git_embedding(model, github_embeddings, batch_size=batch_size)
    cve_embedding = get_cve_embedding(model, test_cve_tuples_list, batch_size=batch_size)



    #accuracy_rate = find_matching_github(model, test_cve, github_embeddings, batch_size=batch_size)

    print(accuracy_rate)

    
    accurate_predictions = 0
    total_predictions = len(test_dataset)

    # Open a file to write the results
    with open('../result_1.txt', 'w') as file:
        for j in range(0, len(test_dataset)):
            cve_embedding, _, _ = test_dataset[j]
            cve_id = test_dataset.cve_df.iloc[j]['CVE_ID']
            if cve_embedding.dim() == 1:
                cve_embedding = cve_embedding.view(1, -1)
            top_k_indices = find_matching_github(model, cve_embedding, test_dataset, device, top_k=1)

            print(top_k_indices)

            # Write the top k indices to the file
            file.write(f"CVE ID: {cve_id}\n")
            for index in top_k_indices:
                top_cve_id = test_dataset.cve_df.iloc[index]['CVE_ID']
                file.write(f"    cves: {top_cve_id}")# Calculate accuracy
            if calculate_accuracy(top_k_indices, cve_id, github_df):
                accurate_predictions += 1

    # Calculate and print the accuracy percentage
    accuracy_percentage = (accurate_predictions / total_predictions) * 100
    print(f"Accuracy: {accuracy_percentage}%")'''








