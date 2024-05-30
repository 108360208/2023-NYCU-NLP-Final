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
    
class KLloss(nn.Module):
    def __init__(self, alpha=0.5):
        super(KLloss, self).__init__()
        self.alpha = alpha

    def forward(self, embedding_reson, embedding, latent_mean, valence, arousal):

        reson_loss = nn.L1Loss(reduction="none")(embedding_reson, embedding)
        pred_valence, pred_arousal = torch.split(latent_mean, 1, dim=2)
        valence_loss = nn.L1Loss(reduction="none")(pred_valence.squeeze(1), valence)
        arousal_loss = nn.L1Loss(reduction="none")(pred_arousal.squeeze(1), arousal)
        # print(reson_loss.mean(), valence_loss.mean(), arousal_loss.mean())
        return reson_loss.mean() + valence_loss.mean() + arousal_loss.mean()

