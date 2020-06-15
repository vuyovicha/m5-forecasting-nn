import torch
import torch.nn as nn
import time
from pinball_loss import PinballLoss
import os
import numpy as np
from pinball_loss import RMSELoss, ValidationRMSELoss, RMSENormalizedLoss
import time


class Trainer(nn.Module):
    def __init__(self, model, data_loader, params, train_amount_of_days, sample_dataset, needed_model_state):
        super(Trainer, self).__init__()
        self.model = model.to(params['device'])
        self.data_loader = data_loader
        self.amount_of_epochs = params['amount_of_epochs']
        self.learning_rate = params['learning_rate']
        self.optimization = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.optimization_step_size = params['optimization_step_size']
        self.gamma_coefficient = params['gamma_coefficient']
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimization, self.optimization_step_size, self.gamma_coefficient)
        self.batch_length = params['batch_size']
        self.output_window_length = params['output_window_length']
        self.train_window_length = train_amount_of_days - 1
        self.sample_dataset = sample_dataset
        self.params = params
        self.val_rmse = ValidationRMSELoss()
        self.clip_value = params['clip_value']

        self.loss_max = 1e8  # validation loss max
        self.loss_epoch = 0
        self.batches = -1
        self.epochs = -1

        if len(needed_model_state) > 0:
            model_save_name = needed_model_state
            path = F"C:/Users/User/Desktop/project 2.0/{model_save_name}"
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimization.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epochs = checkpoint['epoch']
            self.batches = checkpoint['batch']
            self.loss_epoch = checkpoint['loss_epoch']
            self.loss_max = checkpoint['loss_max']
            self.scheduler = checkpoint['scheduler']

    def train_epochs(self):
        for epoch in range(self.amount_of_epochs):
            if epoch >= self.epochs:
                print('EPOCH %d' % epoch)
                loss_epoch = self.train_batches(epoch)
                #if epoch != self.epochs:
                    #self.scheduler.step()
                print('Training_loss: %f' % loss_epoch)
                if not self.params['training_without_val_dataset']:
                    loss_validation = self.validation(epoch)
                    print('Validation_loss: %f' % loss_validation)
                    if loss_validation < self.loss_max:
                        self.loss_max = loss_validation
                else:
                    self.training_without_val_set(epoch)
                print()
        return self.loss_max

    def train_batches(self, epoch):
        self.model.train()
        for batch, (train_dataset, val_dataset, indexes, categories) in enumerate(self.data_loader):
            if batch > self.batches:
                current_start_time = time.time()
                print('Batch %d is here' % batch)
                self.loss_epoch += self.train_batch(train_dataset, val_dataset, indexes, categories)
                print('Batch %d took %5.2f mins' % (batch, ((time.time() - current_start_time) / 60)))

                model_save_name = 'epoch_%d_batch_%d.tar' % (epoch, batch)
                path = F"C:/Users/User/Desktop/project 2.0/{model_save_name}"
                torch.save({
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimization.state_dict(),
                    'epoch': epoch,
                    'batch': batch,
                    'loss_epoch': self.loss_epoch,
                    'loss_max': self.loss_max,
                    'scheduler': self.scheduler
                }, path)

        loss_epoch = self.loss_epoch / (batch + 1)
        self.loss_epoch = 0
        self.batches = -1
        return float(loss_epoch)

    def train_batch(self, train_dataset, val_dataset, indexes, categories):
        self.optimization.zero_grad()
        prediction_values, actual_values = self.model(train_dataset, val_dataset, indexes, categories)
        loss_batch = self.val_rmse(prediction_values, actual_values)
        loss_batch.backward()
        nn.utils.clip_grad_value_(self.model.parameters(), self.clip_value)
        self.optimization.step()
        return float(loss_batch)

    def validation(self, epoch):
        self.model.eval()  # set the model into the evaluation mode
        with torch.no_grad():  # we will not use gradient here
            prediction_values = []
            loss_holdout = 0
            training_rmse_loss = 0
            for batch, (train_dataset, val_dataset, indexes, categories) in enumerate(self.data_loader):
                _, _, holdout_prediction, holdout_actual_values = self.model(train_dataset, val_dataset, indexes, categories, validation=True)
                current_holdout_loss = self.val_rmse(holdout_prediction.float(), holdout_actual_values.float())
                loss_holdout += current_holdout_loss
                prediction_values.append(holdout_prediction)
            predictions = torch.cat([i for i in prediction_values], dim=0)
            self.save_predictions(predictions, epoch)
            loss_holdout = loss_holdout / (batch + 1)
        return float(loss_holdout.detach().cpu().item())

    def training_without_val_set(self, epoch):
        self.model.eval()
        with torch.no_grad():
            prediction_values = []
            for batch, (train_dataset, val_dataset, indexes, categories) in enumerate(self.data_loader):
                holdout_prediction = self.model(train_dataset, val_dataset, indexes, categories, validation=True, training_without_val_dataset=self.params['training_without_val_dataset'])
                prediction_values.append(holdout_prediction)
            predictions = torch.cat([i for i in prediction_values], dim=0)
            self.save_predictions(predictions, epoch)

    def save_predictions(self, predictions, epoch):
        file_name = "Epoch_%d.csv" % epoch
        output_file = self.sample_dataset
        for i in range(len(predictions)):
            for j in range(predictions.shape[1]):
                output_file[i + 1][j + 1] = float(predictions[i][j])
        np.savetxt(file_name, output_file, delimiter=',', fmt="%s")
