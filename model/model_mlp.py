from clip import CLIPModel
from attention import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten(start_dim=1, end_dim=2)

        self.fc1 = nn.Linear(90*1536, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1536)


    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)
        return x

class LSTM_MLP(nn.Module):
    def __init__(self, lstm_hidden_size=8192):
        super(LSTM_MLP, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=1536, hidden_size=lstm_hidden_size, batch_first=True)

        # MLP layers
        self.fc1 = nn.Linear(lstm_hidden_size, 8192)
        self.fc2 = nn.Linear(8192, 4096)
        self.fc3 = nn.Linear(4096, 2048)
        self.fc4 = nn.Linear(2048, 1536)

    def forward(self, x):
        # LSTM layer
        lstm_out, (hidden, _) = self.lstm(x) #(32,90,8192)

        # Use the final hidden state as input to the MLP
        x = hidden[-1] #(32,8192) accesses the last layer's hidden state

        # MLP layers
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x)

        return x

class CombinedModel(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dim_head, d_output, temperature=0.7, dropout=0.1):
        super().__init__()
        self.cross_attention = BidirectionalCrossAttention(n_head, d_model, dim_head)
        '''self.fc1 = nn.Linear(d_model, d_model)  # First dense layer
        self.fc2 = nn.Linear(d_model, d_model)  # Second dense layer
        self.norm = nn.LayerNorm(d_model)'''
        self.multi_head_attention_cve = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)
        self.multi_head_attention_diff = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)
        self.multi_head_attention_message = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)
        self.mlp = LSTM_MLP()


        self.clip_model = CLIPModel(temperature)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, cve_embeddings, github_message, github_diff):
        github_message = self.multi_head_attention_message(github_message)
        github_diff = self.multi_head_attention_diff(github_diff)
        projected_github_embeddings = self.cross_attention(github_message, github_diff, github_diff)
        '''github_embeddings = self.fc1(attn_output)
        github_embeddings = self.norm(github_embeddings)  # Apply normalization
        github_embeddings = F.relu(github_embeddings)
        github_embeddings = self.fc2(github_embeddings)'''
        #pooled_github_embeddings = self.global_max_pool(projected_github_embeddings.transpose(1, 2)).squeeze(-1)
        pooled_github_embeddings = self.mlp(projected_github_embeddings)

        projected_cve_embeddings = self.multi_head_attention_cve(cve_embeddings)
        pooled_cve_embeddings = projected_cve_embeddings.squeeze(1)

        '''pooled_cve_embeddings_norm = F.normalize(pooled_cve_embeddings)
        pooled_github_embeddings_norm = F.normalize(pooled_github_embeddings)'''
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        target = torch.ones(cve_embeddings.size(0),device=device)
        cosine_loss = nn.CosineEmbeddingLoss().to(device)

        loss = cosine_loss(pooled_cve_embeddings, pooled_github_embeddings, target)

        #loss = self.clip_model(pooled_github_embeddings,pooled_cve_embeddings)
        return loss,pooled_cve_embeddings,pooled_github_embeddings

    def predict(self, cve_embeddings, github_embeddings, top_n=5):
        # Ensure the model is in evaluation mode

        with torch.no_grad():
            correct_predictions = 0
            total_predictions = len(cve_embeddings)

            for i,cve_embedding in enumerate(cve_embeddings.split(1, dim=0)):
                similarities = []
                for github in github_embeddings.split(1, dim=0):
                    similarity = torch.matmul(cve_embedding, github.T)
                    similarities.append(similarity.item())

                # Compute cosine similarity
                top_n_indices = np.argsort(similarities)[-top_n:]
                #top_n_matches = [(pooled_github_embeddings_norm[i], similarities[i]) for i in top_n_indices]

                if i in top_n_indices:
                    correct_predictions += 1

        acc = correct_predictions / total_predictions

        return acc

    def predict_single_cve(self, cve_embedding, all_github_embeddings,top_n):
        self.eval()

        with torch.no_grad():
            # Project cve_embedding using multi-head attention
            projected_cve_embedding = self.multi_head_attention_cve(cve_embedding)

            # Calculate similarities with all GitHub embeddings
            similarities = torch.matmul(projected_cve_embedding, all_github_embeddings.T)

            # Get top 5 most similar GitHub embeddings
            top_5_similarities, top_5_indices = torch.topk(similarities, top_n)

        return top_5_indices.squeeze(0)