from clip import CLIPModel
from attention import *
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import torch
import torch.nn as nn




class LSTM(nn.Module):
    def __init__(self, lstm_hidden_size):
        super(LSTM, self).__init__()
        # LSTM layer
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, batch_first=True, bidirectional=True)
        #self.lstm = nn.LSTM(input_size=1536, hidden_size=lstm_hidden_size, batch_first=True)

        # MLP layers
        #self.fc1 = nn.Linear(lstm_hidden_size, 256)


    def forward(self, x):
        # LSTM layer
        lstm_out, (hidden, _) = self.lstm(x) #(32,90,256)

        # Use the final hidden state as input to the MLP
        x = hidden[-1] #(32,8192) accesses the last layer's hidden state

        # MLP layers
        #x = torch.relu(self.fc1(x))

        return x

class CombinedModel(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, d_output, temperature=0.7, dropout=0.1):
        super().__init__()
        self.cross_attention = BidirectionalCrossAttention(16, 256, 16)
        '''self.fc1 = nn.Linear(d_model, d_model)  # First dense layer
        self.fc2 = nn.Linear(d_model, d_model)  # Second dense layer
        self.norm = nn.LayerNorm(d_model)'''
        self.multi_head_attention_cve = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)
        self.multi_head_attention_diff = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)
        self.multi_head_attention_message = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)
        self.down_sample_1 = MultiHeadAttention_down(16, 256, 16, 16, 256, dropout)
        self.down_sample_2 = MultiHeadAttention_down(16, 256, 16, 16, 256, dropout)
        self.down_sample_3 = MultiHeadAttention_down(16, 256, 16, 16, 256, dropout)
        self.pooling = nn.AdaptiveMaxPool1d(256)
        self.lstm = LSTM()




        self.clip_model = CLIPModel(temperature)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, cve_embeddings, github_message, github_diff):
        github_message = self.pooling(github_message)
        github_diff = self.pooling(github_diff)
        #github_diff = self.multi_head_attention_diff(github_diff)
        #github_diff_a = self.pooling(github_diff)
        projected_github_embeddings = self.cross_attention(github_message, github_diff, github_diff)
        #pooled_github_embeddings = self.global_max_pool(projected_github_embeddings.transpose(1, 2)).squeeze(-1)
        pooled = self.down_sample_1(projected_github_embeddings) # (32,45,1536)
        pooled = self.down_sample_2(pooled) # (32,23,1536)
        pooled = self.down_sample_3(pooled)
        pooled_github_embeddings = self.lstm(pooled)
        #pooled_github_embeddings = self.global_max_pool(pooled.transpose(1, 2)).squeeze(-1)
        #pooled_github_embeddings = self.pooling(pooled_github_embeddings)
        pooled_github_embeddings = pooled_github_embeddings.squeeze(1)



        #projected_cve_embeddings = self.multi_head_attention_cve(cve_embeddings)
        projected_cve_embeddings = self.pooling(cve_embeddings)
        pooled_cve_embeddings = projected_cve_embeddings.squeeze(1)
        #pooled_cve_embeddings = self.pooling(cve_embeddings)

        pooled_cve_embeddings_norm = F.normalize(pooled_cve_embeddings)
        pooled_github_embeddings_norm = F.normalize(pooled_github_embeddings)


        loss = self.clip_model(pooled_github_embeddings_norm,pooled_cve_embeddings_norm)
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
            #dot_similarity = text_embeddings_n @ image_embeddings_n.T
            #values, indices = torch.topk(dot_similarity.squeeze(0), n)

            # Get top 5 most similar GitHub embeddings
            top_5_similarities, top_5_indices = torch.topk(similarities, top_n)

        return top_5_indices.squeeze(0)