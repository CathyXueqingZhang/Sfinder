from clip import CLIPModel
from attention import *


class CombinedModel(nn.Module):
    def __init__(self, n_head, d_model, d_k, d_v, dim_head, d_output, temperature=0.7, dropout=0.1):
        super().__init__()
        self.cross_attention = BidirectionalCrossAttention(n_head, d_model, dim_head)
        self.fc1 = nn.Linear(d_model, d_model)  # First dense layer
        self.fc2 = nn.Linear(d_model, d_model)  # Second dense layer
        self.norm = nn.LayerNorm(d_model)
        self.multi_head_attention_cve = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)
        self.multi_head_attention_git = MultiHeadAttention(n_head, d_model, d_k, d_v, d_output, dropout)


        self.clip_model = CLIPModel(temperature)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, cve_embeddings, github_message, github_diff):
        attn_output = self.cross_attention(github_message,github_diff)
        github_embeddings = self.fc1(attn_output)
        github_embeddings = self.norm(github_embeddings)  # Apply normalization
        github_embeddings = F.relu(github_embeddings)
        github_embeddings = self.fc2(github_embeddings)
        projected_github_embeddings = self.multi_head_attention_git(github_embeddings)
        pooled_github_embeddings = self.global_max_pool(projected_github_embeddings.transpose(1, 2)).squeeze(-1)

        projected_cve_embeddings = self.multi_head_attention_cve(cve_embeddings)
        pooled_cve_embeddings = self.global_max_pool(projected_cve_embeddings.transpose(1, 2)).squeeze(-1)

        loss = self.clip_model(pooled_github_embeddings, pooled_cve_embeddings)
        return loss

    def predict(self, cve_embeddings, github_message, github_diff):
        # Ensure the model is in evaluation mode
        self.eval()

        with torch.no_grad():
            # Process github_message and github_diff with cross attention and subsequent layers
            attn_output = self.cross_attention(github_message, github_diff, github_diff)
            github_embeddings = self.fc1(attn_output)
            github_embeddings = self.norm(github_embeddings)
            github_embeddings = F.relu(github_embeddings)
            github_embeddings = self.fc2(github_embeddings)
            projected_github_embeddings = self.multi_head_attention_git(github_embeddings)
            pooled_github_embeddings = self.global_max_pool(projected_github_embeddings.transpose(1, 2)).squeeze(-1)

            # Project cve_embeddings using multi-head attention
            projected_cve_embeddings = self.multi_head_attention_cve(cve_embeddings)
            pooled_cve_embeddings = self.global_max_pool(projected_cve_embeddings.transpose(1, 2)).squeeze(-1)

            # Calculate similarities
            similarities = torch.matmul(pooled_cve_embeddings, pooled_github_embeddings.T)
            most_similar_indices = torch.argmax(similarities, dim=1)

        return most_similar_indices