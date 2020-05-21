import torch
import torch.nn as nn
import time
from pinball_loss import PinballLoss
import os
import numpy as np
from pinball_loss import RMSELoss


class Trainer(nn.Module):
    def __init__(self, model, data_loader, params, train_amount_of_days, sample_dataset, real_values_starting_indexes):
        super(Trainer, self).__init__()
        self.model = model.to(params['device'])
        self.data_loader = data_loader
        self.amount_of_epochs = params['amount_of_epochs']
        self.learning_rate = params['learning_rate']
        self.optimization = torch.optim.Adam(self.model.parameters(), self.learning_rate)  # do you remember nn.Parameter? https://arxiv.org/pdf/1412.6980.pdf
        self.optimization_step_size = params['optimization_step_size']
        self.gamma_coefficient = params['gamma_coefficient']
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimization, self.optimization_step_size, self.gamma_coefficient)
        self.epochs = 0
        self.training_percentile = params['training_percentile']
        self.training_tau = self.training_percentile / 100
        self.batch_length = params['batch_size']
        self.output_window_length = params['output_window_length']
        self.measure_pinball_val = PinballLoss(self.training_tau, self.output_window_length * self.batch_length, params['device'])
        self.train_window_length = train_amount_of_days - 1
        self.measure_pinball_train = PinballLoss(self.training_tau, self.train_window_length * self.batch_length, params['device'])
        self.clip_value = params['clip_value']  # todo
        self.sample_dataset = sample_dataset
        self.params = params
        self.real_values_starting_indexes = real_values_starting_indexes
        self.measure_rmse = RMSELoss(self.real_values_starting_indexes)

    def train_epochs(self):
        loss_max = 1e8
        #time_start = time.time()
        for epoch in range(self.amount_of_epochs):
            print('Epoch % is going live' % epoch)
            loss_epoch = self.train_batches()
            self.scheduler.step()  # changed the place of this scheduler (it adjusts the learning rate)
            if loss_epoch < loss_max:  # can we save just model parameters?
                #self.save_current_model()  # todo uncomment
                loss_max = loss_epoch  # isn't it?
            print('Training_loss: %f' % loss_epoch)
            if not self.params['training_without_val_dataset']:
                loss_validation = self.validation()
                print('Validation_loss: %f' % loss_validation)
            else:
                self.training_without_val_set()
            # we can save current model losses here
            print()
            # print time here
        return loss_max  # should we know parameters and what not to generate predictions?

    def train_batches(self):
        self.model.train()
        loss_epoch = 0
        for batch, (train_dataset, val_dataset, indexes, categories) in enumerate(self.data_loader):
            print('Batch %d is here' % batch)
            loss_epoch += self.train_batch(train_dataset, val_dataset, indexes, categories)
        loss_epoch = loss_epoch / (batch + 1)
        self.epochs += 1

        # some log hists and what not are here

        return float(loss_epoch)

    def train_batch(self, train_dataset, val_dataset, indexes, categories):  # removed mean square log difference from the model output
        self.optimization.zero_grad()
        prediction_values, actual_values = self.model(train_dataset, val_dataset, indexes, categories)  # , _, _, _, _
        #prediction_values, actual_values, _, _, _, _ = self.model(train_dataset, val_dataset, indexes, categories, validation=True)  # todo
        #loss_batch = self.measure_pinball_train(prediction_values, actual_values)
        loss_batch = self.measure_rmse(prediction_values, actual_values, indexes)
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
            prediction_values = []
            loss_holdout = 0
            for batch, (train_dataset, val_dataset, indexes, categories) in enumerate(self.data_loader):
                _, _, holdout_prediction, holdout_output_cat, holdout_actual_values, holdout_actual_values_deseasonalized_normalized = self.model(
                    train_dataset, val_dataset, indexes, categories, validation=True)
                # TODO we should measure holdout_output last value with actual deseas and norm values
                current_holdout_loss = self.measure_pinball_val(holdout_prediction.float(), holdout_actual_values.float())
                #current_holdout_loss = self.measure_rmse(holdout_output_cat.float(), holdout_actual_values_deseasonalized_normalized.float())  # todo THIS
                #current_holdout_loss = self.measure_pinball_val(holdout_output_cat.float(), holdout_actual_values_deseasonalized_normalized.float())
                loss_holdout += current_holdout_loss
                prediction_values.append(holdout_prediction)
            predictions = torch.cat([i for i in prediction_values], dim=0)
            self.save_predictions(predictions)
            loss_holdout = loss_holdout / (batch + 1)
        return float(loss_holdout.detach().cpu().item())

    def training_without_val_set(self):
        self.model.eval()
        with torch.no_grad():
            prediction_values = []
            for batch, (train_dataset, val_dataset, indexes, categories) in enumerate(self.data_loader):
                holdout_prediction = self.model(train_dataset, val_dataset, indexes, categories, validation=True, training_without_val_dataset=self.params['training_without_val_dataset'])
                prediction_values.append(holdout_prediction)
            predictions = torch.cat([i for i in prediction_values], dim=0)
            self.save_predictions(predictions)

    def save_predictions(self, predictions):
        file_name = "Epoch_%d.csv" % self.epochs
        output_file = self.sample_dataset
        for i in range(len(predictions)):
            for j in range(predictions.shape[1]):
                output_file[i + 1][j + 1] = float(predictions[i][j])
        np.savetxt(file_name, output_file, delimiter=',', fmt="%s")











