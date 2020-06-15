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


class RMSELoss(torch.nn.Module):
    def __init__(self, real_values_starting_indexes):
        super(RMSELoss, self).__init__()
        self.real_values_starting_indexes = real_values_starting_indexes

    def forward(self, predictions, actuals, indexes):
        predictions_reshape_list = []
        actuals_reshape_list = []
        for i in range(predictions.shape[1]):
            predictions_reshape_list.append(torch.cat([batch_values[i].unsqueeze(0) for batch_values in predictions], dim=0))
            actuals_reshape_list.append(torch.cat([batch_values[i].unsqueeze(0) for batch_values in actuals], dim=0))
        predictions_reshape = torch.cat([i.unsqueeze(0) for i in predictions_reshape_list], dim=0)
        actuals_reshape = torch.cat([i.unsqueeze(0) for i in actuals_reshape_list], dim=0)
        loss = 0
        amount_of_values = 0
        for i in range(len(predictions_reshape)):
            for j in range(len(predictions_reshape[i])):
                if j >= self.real_values_starting_indexes[indexes[i]]:
                    loss += (predictions_reshape[i, j] - actuals_reshape[i, j]) ** 2.0
                    amount_of_values += 1
        return torch.sqrt(loss / amount_of_values)


class RMSENormalizedLoss(torch.nn.Module):
    def __init__(self, real_values_starting_indexes):
        super(RMSENormalizedLoss, self).__init__()
        self.real_values_starting_indexes = real_values_starting_indexes

    def forward(self, predictions, actuals, indexes):
        loss = 0
        amount_of_values = 0
        for i in range(len(predictions)):
            for j in range(len(predictions[i])):
                if j >= self.real_values_starting_indexes[indexes[i]]:
                    loss += (predictions[i, j] - actuals[i, j]) ** 2.0
                    amount_of_values += 1
        return torch.sqrt(loss / amount_of_values)


class ValidationRMSELoss(torch.nn.Module):
    def __init__(self):
        super(ValidationRMSELoss, self).__init__()

    def forward(self, predictions, actuals):
        criterion = nn.MSELoss()
        loss = torch.sqrt(criterion(predictions, actuals))
        return loss



