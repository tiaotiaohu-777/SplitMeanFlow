import torch
import torch.nn.functional as F
import numpy as np

class SplitMeanFlow:
    def __init__(self):
        pass

    def sample_time(self, B, device):
        t = torch.rand(B, device=device)
        r = torch.rand(B, device=device)
        t, r = torch.minimum(t, r), torch.maximum(t, r)
        return t, r

    def BoundaryLoss(self, student, teacher, x, c):
        B = x.shape[0]
        device = x.device

        t = torch.rand(B, device=device)
        t_exp = t.view(-1,1,1,1)

        eps = torch.randn_like(x)
        z_t = (1 - t_exp) * eps + t_exp * x

        with torch.no_grad():
            v_teacher = teacher(z_t, t, c)

        pred = student(z_t, t, c, t)
        loss_mse = F.mse_loss(pred, v_teacher)

        return loss_mse

    def SplitLoss(self, student, ema, x, c):
        B = x.shape[0]
        device = x.device

        t, r = self.sample_time(B, device)

        t_exp = t.view(-1,1,1,1)
        r_exp = r.view(-1,1,1,1)

        eps = torch.randn_like(x)
        z_t = (1 - t_exp) * eps + t_exp * x

        lam = torch.rand(B, device=device)
        s = (1 - lam) * t + lam * r

        s_exp = s.view(-1,1,1,1)

        with torch.no_grad():
            u_2 = ema(z_t, t, c, s)
            z_s = z_t + (s_exp - t_exp) * u_2

            u_1 = ema(z_s, s, c, r)

            target = (1 - lam.view(-1,1,1,1)) * u_1 + lam.view(-1,1,1,1) * u_2

        pred = student(z_t, t, c, r)
        loss_mse = F.mse_loss(pred, target)

        return loss_mse