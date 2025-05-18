import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning as used in SimCLR
    Implementation based on "A Simple Framework for Contrastive Learning of Visual Representations"
    
    Paper reference: https://arxiv.org/abs/2002.05709
    """
    def __init__(self, temperature=0.07, batch_size=None):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, z_i, z_j=None):
        """
        Calculate NT-Xent loss between embedding pairs
        
        Args:
            z_i: First set of embeddings [batch_size, embedding_dim] (or [batch_size*2, embedding_dim] if z_j is None)
            z_j: Optional second set of embeddings [batch_size, embedding_dim]
            
        Returns:
            NT-Xent loss value
        """
        # If only one input is provided, split it in half (assumes augmented pairs are stacked)
        if z_j is None:
            batch_size = self.batch_size or z_i.shape[0] // 2
            z_i, z_j = z_i[:batch_size], z_i[batch_size:]
        else:
            batch_size = self.batch_size or z_i.shape[0]
            
        # Normalize embeddings along feature dimension
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)
        
        # Concatenate embeddings from the two views
        representations = torch.cat([z_i, z_j], dim=0)
        
        # Compute similarity matrix
        similarity_matrix = torch.exp(torch.mm(representations, representations.t()) / self.temperature)
        
        # Mask out self-similarity
        mask = ~torch.eye(2 * batch_size, dtype=bool, device=z_i.device)
        similarity_matrix = similarity_matrix.masked_select(mask).view(2 * batch_size, -1)
        
        # Create positive pair mask (augmented pairs should be similar)
        pos_mask = torch.zeros((2 * batch_size, 2 * batch_size - 1), dtype=bool, device=z_i.device)
        pos_mask[:batch_size, batch_size-1:2*batch_size-1] = torch.eye(batch_size, device=z_i.device)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, device=z_i.device)
        
        # Extract positive and negatives
        positives = similarity_matrix.masked_select(pos_mask).view(2 * batch_size, 1)
        negatives = similarity_matrix.masked_select(~pos_mask).view(2 * batch_size, -1)
        
        # Concatenate positive with negatives for log softmax calculation
        logits = torch.cat([positives, negatives], dim=1)
        
        # Create labels (positive is always at index 0)
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=z_i.device)
        
        # Calculate cross entropy loss
        return self.criterion(logits, labels)


class MorphoFeaturesLoss(nn.Module):
    """
    Combined loss function as described in the MorphoFeatures paper:
    L_total = L_NT-Xent + λ_AE·L_MSE + λ_norm·‖z‖₂
    
    For shape branch: contrastive only (λ_AE = 0)
    For texture branches: all three terms
    """
    def __init__(self, temperature=0.07, lambda_ae=1.0, lambda_norm=0.01, batch_size=None):
        super(MorphoFeaturesLoss, self).__init__()
        self.contrastive_loss = NTXentLoss(temperature=temperature, batch_size=batch_size)
        self.reconstruction_loss = nn.MSELoss(reduction='mean')
        self.lambda_ae = lambda_ae
        self.lambda_norm = lambda_norm
        
    def forward(self, output, target=None):
        """
        Calculate combined loss
        
        Args:
            output: Tuple containing model outputs (projection, embedding, reconstruction if texture)
            target: Input volume for reconstruction loss (only for texture branches)
            
        Returns:
            Combined loss value
        """
        # Unpack model outputs
        if len(output) == 3:
            # Texture branch (projection, embedding, reconstruction)
            projection, embedding, reconstruction = output
            
            # Check if target is a list/tuple
            if isinstance(target, (list, tuple)):
                target = target[0]
                
            # Contrastive loss
            contrastive = self.contrastive_loss(projection)
            
            # Reconstruction loss (autoencoder)
            if self.lambda_ae > 0 and target is not None and reconstruction is not None:
                reconstruction_loss = self.reconstruction_loss(reconstruction, target)
            else:
                reconstruction_loss = torch.tensor(0.0, device=projection.device)
            
            # L2 norm regularization
            norm_loss = torch.norm(embedding, p=2, dim=1).mean()
            
            # Combined loss
            return contrastive + self.lambda_ae * reconstruction_loss + self.lambda_norm * norm_loss
            
        else:
            # Shape branch (projection, embedding) - contrastive only
            projection, embedding = output
            
            # Contrastive loss
            contrastive = self.contrastive_loss(projection)
            
            # L2 norm regularization (still keep this term)
            norm_loss = torch.norm(embedding, p=2, dim=1).mean()
            
            return contrastive + self.lambda_norm * norm_loss


# Default configurations used in the paper
def get_shape_loss():
    """Get loss function for shape branch (contrastive only)"""
    return MorphoFeaturesLoss(lambda_ae=0.0, lambda_norm=0.01)

def get_texture_loss():
    """Get loss function for texture branches (contrastive + reconstruction + regularization)"""
    return MorphoFeaturesLoss(lambda_ae=1.0, lambda_norm=0.01)


# For testing
if __name__ == "__main__":
    # Test NT-Xent loss
    batch_size = 8
    embedding_dim = 80
    
    # Create random embeddings for testing
    embeddings = torch.randn(batch_size * 2, embedding_dim)
    
    # Test NTXentLoss
    nt_loss = NTXentLoss()
    loss_value = nt_loss(embeddings)
    print(f"NT-Xent Loss: {loss_value.item()}")
    
    # Test combined loss for shape branch
    shape_loss = get_shape_loss()
    loss_value = shape_loss((embeddings[:batch_size], embeddings[:batch_size]))
    print(f"Shape Loss: {loss_value.item()}")
    
    # Test combined loss for texture branch
    texture_loss = get_texture_loss()
    input_volumes = torch.randn(batch_size, 1, 32, 32, 32)
    reconstructed = torch.randn(batch_size, 1, 32, 32, 32)
    loss_value = texture_loss((embeddings[:batch_size], embeddings[:batch_size], reconstructed), input_volumes)
    print(f"Texture Loss: {loss_value.item()}") 