import torch
import torch.nn as nn
import time
from pinball_loss import PinballLoss



class Trainer(nn.Module):
    def __init__(self, model, data_loader,params):
        super(Trainer, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.amount_of_epochs = params['amount_of_epochs']#16
        self.learning_rate = params['learning_rate']#1e-3
        self.optimization = torch.optim.Adam(self.model.parameters(),
                                             self.learning_rate)  # do you remember nn.Parameter? https://arxiv.org/pdf/1412.6980.pdf
        self.optimization_step_size = params['optimization_step_size']#5
        self.gamma_coefficient = params['gamma_coefficient']#0.5
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimization, self.optimization_step_size,
                                                         self.gamma_coefficient)

        self.epochs = 0

        self.training_percentile = params['training_percentile']#45
        self.training_tau = self.training_percentile / 100
        self.batch_length = 1024
        self.output_window_length = 28
        self.measure_pinball = PinballLoss(self.training_tau, self.output_window_length * self.batch_length,
                                           'cpu')  # last parameter - device CHECK
        self.clip_value = params['clip_value']#23  # todo

    def train_epochs(self):
        loss_max = 1e8
        time_start = time.time()
        for epoch in range(self.amount_of_epochs):
            print('Epoch % is going live' % epoch)
            loss_epoch = self.train_batches()
            self.scheduler.step()  # changed the place of this scheduler (it adjusts the learning rate)
            if loss_epoch < loss_max:  # can we save just model parameters?
                self.save_current_model()
                loss_max = loss_epoch  # isn't it?
            loss_validation = self.validation()
            # we can save current model losses here
        print('Training has taken: %5.1f minutes' % (time.time() - time_start) / 60)
        return loss_max

    def train_batches(self):
        self.model.train()
        loss_epoch = 0
        for batch, (train_dataset, val_dataset, indexes) in enumerate(self.data_loader):
            print('Epoch % is here' % batch)
            loss_epoch += self.train_batch(train_dataset, val_dataset, indexes)
        loss_epoch = loss_epoch / (batch + 1)
        self.epochs += 1

        # some log hists and what not are here

        return loss_epoch

    def train_batch(self, train_dataset, val_dataset, indexes):  # removed mean square log difference from the model output
        self.optimization.zero_grad()
        prediction_values, actual_values, _, _, _, _ = self.model(train_dataset, val_dataset, indexes)
        loss_batch = self.measure_pinball(prediction_values, actual_values)
        loss_batch.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
        self.optimization.step()
        return float(loss_batch)

    def save_current_model(self):  # can we just save the output? or we can save the model parameters and compute in the end
        directory = "...."  # create a directory somewhere
        path_file = os.path.join(directory, 'models', time.time())  # changed the second parameter
        path_model = os.path.join(path_file, 'current_model_{}.py'.format(self.epochs))  # py or pyt?
        os.makedirs(path_file, exist_ok=True)
        torch.save(self.model.state_dict(), path_model)

    def validation(self):
        self.model.eval()  # set the model into the evaluation mode
        with torch.no_grad():  # we will not use gradient here
            actual_values = []
            prediction_values = []
            # todo: also info cat is here - learn what is that

            loss_holdout = 0
            for batch, (train_dataset, val_dataset, indexes) in enumerate(self.data_loader):
                _, _, holdout_prediction, holdout_output, holdout_actual_values, holdout_actual_values_deseasonalized_normalized = self.model(
                    train_dataset, val_dataset, indexes)
                loss_holdout += self.measure_pinball(holdout_output.unsqueeze(0).float(),
                                                     holdout_actual_values_deseasonalized_normalized.unsqueeze(
                                                         0).float())
                prediction_values.extend(holdout_prediction.view(-1).cpu().detach().numpy())
                actual_values.extend(holdout_actual_values.view(-1).cpu().detach().numpy())  # what is this?
                # infocat
            loss_holdout = loss_holdout / (batch + 1)
        return loss_holdout.detach().cpu().item()











