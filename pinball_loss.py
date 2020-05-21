import torch
import torch.nn as nn


class PinballLoss(nn.Module):
    def __init__(self, training_tau, output_size, device):
        super(PinballLoss, self).__init__()
        self.training_tau = training_tau
        self.output_size = output_size
        self.device = device

    def forward(self, predictions, actuals):
        cond = torch.zeros_like(predictions).to(self.device)
        loss = torch.sub(actuals, predictions).to(self.device)

        less_than = torch.mul(loss, torch.mul(torch.gt(loss, cond).type(torch.FloatTensor).to(self.device),
                                              self.training_tau))

        greater_than = torch.mul(loss, torch.mul(torch.lt(loss, cond).type(torch.FloatTensor).to(self.device),
                                                 (self.training_tau - 1)))

        final_loss = torch.add(less_than, greater_than)
        return torch.sum(final_loss) / self.output_size * 2
