from torch import nn, optim
import torch.nn.functional as F


class CLIPModel(nn.Module):
    def __init__(self, temperature):
        super().__init__()
        self.temperature = temperature

    def cross_entropy(self, preds, targets, reduction='none'):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        if reduction == "none":
            return loss
        elif reduction == "mean":
            return loss.mean()

    def forward(self, github_embeddings, cve_embeddings):
        # Calculating the Loss
        logits = (cve_embeddings @ github_embeddings.T)

        github_similarity = github_embeddings @ github_embeddings.T
        cve_similarity = cve_embeddings @ cve_embeddings.T
        targets = F.softmax(
            (github_similarity + cve_similarity) / 2 / self.temperature, dim=-1
        )
        cve_loss = self.cross_entropy(logits, targets, reduction='none')
        github_loss = self.cross_entropy(logits.T, targets.T, reduction='none')
        loss =  (github_loss + cve_loss) / 2.0 # shape: (batch_size)
        return loss.mean()

