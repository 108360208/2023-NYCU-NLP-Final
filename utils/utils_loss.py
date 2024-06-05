import torch
import torch.nn as nn
from scipy.stats import pearsonr

class PearsonLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(PearsonLoss, self).__init__()
        self.alpha = alpha

    def forward(self, outputs, valence, arousal, valence_sd, arousal_ad ):
        pred_valence, pred_arousal = torch.split(outputs, 1, dim=1)
      
        
        # 分别计算 valence 和 arousal 的 L1 损失
        l1_loss_valence = nn.L1Loss(reduction="none")(pred_valence, valence) * valence_sd
        l1_loss_arousal = nn.L1Loss(reduction="none")(pred_arousal, arousal) * arousal_ad
        
        # l1_loss_valence[valence_sd == 0] = 0.1  # 将标准差为0的位置的损失设置为0
        # l1_loss_arousal[arousal_ad == 0] = 0.1  # 将标准差为0的位置的损失设置为0
        l1_loss_valence = l1_loss_valence.mean()
        l1_loss_arousal = l1_loss_arousal.mean()
        # arousal_l1_loss = nn.L1Loss()(pred_arousal.float(), arousal)
        # print(pred_valence.float().squeeze().cpu().detach().numpy())
        # print(valence.cpu().detach().numpy())
        valence_pearson = pearsonr(pred_valence.float().squeeze().cpu().detach().numpy(), valence.squeeze().cpu().detach().numpy())[0]
        arousal_pearson = pearsonr(pred_valence.float().squeeze().cpu().detach().numpy(), arousal.squeeze().cpu().detach().numpy())[0]
        combined_loss = l1_loss_valence + l1_loss_arousal
        return combined_loss
    
class Reconstloss(nn.Module):
    def __init__(self, alpha=0.5):
        super(Reconstloss, self).__init__()
        self.alpha = alpha

    def forward(self, embedding_recon, embedding, latent_mean, valence, arousal):

        reson_loss = nn.L1Loss(reduction="none")(embedding_recon, embedding)
        pred_valence, pred_arousal = torch.split(latent_mean, 1, dim=2)
        valence_loss = nn.L1Loss(reduction="none")(pred_valence.squeeze(1), valence)
        arousal_loss = nn.L1Loss(reduction="none")(pred_arousal.squeeze(1), arousal)
        return reson_loss.mean() 
class KL_Loss(nn.Module):
    def __init__(self, alpha=0.5):
        super(KL_Loss, self).__init__()
        self.alpha = alpha

    def forward(self, mu1, logvar1, mu2, logvar2, batch_size):
        # print(mu1, logvar1, mu2, logvar2)
        kl_loss = kl_divergence(mu1, logvar1, mu2, logvar2)
        
        return kl_loss / batch_size
    
def kl_criterion(mu, logvar, batch_size):
  KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
  KLD /= batch_size  
  return KLD * 0.1


def kl_divergence(mu1, logvar1, mu2, logvar2):

    var1 = torch.exp(logvar1)
    var2 = torch.exp(logvar2)

    kl = 0.5 * (logvar2 - logvar1 + (var1 + (mu1 - mu2).pow(2)) / var2 - 1)
    loss = kl.sum()
    
    return loss

