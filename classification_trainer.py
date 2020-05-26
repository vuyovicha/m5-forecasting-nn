import torch
import torch.nn as nn
import classification_preprocessing
from dataset import ClassifierValDatset
from torch.utils.data import DataLoader
import numpy as np


class ClassifierTrainer(nn.Module):
    def __init__(self, model, train_data_loader, initial_val_data_loader, params, val_targets, prices_dataset, encoded_categories, preprocessed_time_categories, snap_categories_numerical, starting_validation_day, last_day_zero_indexes, is_more_zeros_than_threshold_list):
        super(ClassifierTrainer, self).__init__()
        self.model = model.to(params['device'])
        self.train_data_loader = train_data_loader
        self.val_data_loader = initial_val_data_loader
        self.learning_rate = params['learning_rate']
        self.optimization = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.amount_of_epochs = params['amount_of_epochs']
        self.epochs = 0
        self.criterion = nn.BCEWithLogitsLoss()
        self.params = params
        self.val_targets = val_targets
        self.prices_dataset = prices_dataset
        self.encoded_categories = encoded_categories
        self.preprocessed_time_categories = preprocessed_time_categories
        self.snap_categories_numerical = snap_categories_numerical
        self.starting_validation_day = starting_validation_day
        self.last_day_zero_indexes = last_day_zero_indexes
        self.is_more_zeros_than_threshold_list = is_more_zeros_than_threshold_list

    def train_epochs(self):
        loss_max = 1e8
        for epoch in range(self.amount_of_epochs):
            print('Epoch % is going live' % epoch)
            loss_epoch = self.train_batches()
            if loss_epoch < loss_max:
                loss_max = loss_epoch
            print('Training_loss: %f' % loss_epoch)
            if not self.params['training_without_val_dataset']:
                loss_validation, accuracy = self.validation()
                print('Validation_loss: %f' % loss_validation)
                print('Accuracy: %f' % accuracy)
            else:
                self.training_without_val_set()
            print()
        return loss_max

    def train_batches(self):
        self.model.train()
        loss_epoch = 0
        total_values = 0
        correct_overall = 0
        for batch, (series_index, day_index, numerical_data, categorical_data, target, index) in enumerate(self.train_data_loader):
            #print('Batch %d is here' % batch)
            current_loss, current_values, correct_values = self.train_batch(series_index, day_index, numerical_data, categorical_data, target, index)
            correct_overall += correct_values
            loss_epoch += current_values * current_loss
            total_values += current_values
        self.epochs += 1
        accuracy = float(correct_overall) / total_values
        print("TRAINING_ACCURACY: %f" % accuracy)
        return float(loss_epoch) / total_values

    def train_batch(self, series_index, day_index, numerical_data, categorical_data, target, index):
        batch_size = target.shape[0]
        #print(numerical_data)
        output = self.model(numerical_data, categorical_data)
        loss_batch = self.criterion(output, target.unsqueeze(1))
        self.optimization.zero_grad()
        loss_batch.backward()
        self.optimization.step()
        rounded_output = torch.round(torch.sigmoid(output))  # added sigmoid here
        correct_values = (rounded_output == target.unsqueeze(1)).float().sum().item()
        return float(loss_batch), batch_size, float(correct_values)

    def validation(self):
        self.model.eval()
        with torch.no_grad():
            prediction_values = []
            loss_holdout = 0
            correct_values = 0
            total_values = 0
            starting_validation_day = self.starting_validation_day
            for i in range(self.params['output_window_length']):
                current_predictions = []
                for batch, (series_index, day_index, numerical_data, categorical_data, index) in enumerate(self.val_data_loader):
                    batch_size = len(self.val_targets[:, i])
                    print(numerical_data)
                    #print(categorical_data)  # todo problem with last day zero index
                    output = self.model(numerical_data, categorical_data)
                    #print(output)
                    starting_index = int(series_index[0].cpu().numpy())
                    ending_index = int(series_index[-1].cpu().numpy()) + 1
                    numpy_targets = self.val_targets[starting_index:ending_index, i]
                    targets = [torch.tensor(numpy_targets[j]) for j in range(len(numpy_targets))]
                    stack_targets = torch.stack([j for j in targets])
                    stack_targets = stack_targets.type_as(output)
                    current_holdout_loss = self.criterion(output, stack_targets.unsqueeze(1))
                    loss_holdout += batch_size * current_holdout_loss
                    total_values += batch_size
                    rounded_output = torch.round(torch.sigmoid(output))  # added sigmoid here
                    rounded_output = classification_preprocessing.compute_zero_prices_for_val(self.prices_dataset, rounded_output, starting_validation_day)
                    #print(rounded_output)
                    current_predictions.append(rounded_output)
                    correct_values += (rounded_output == stack_targets.unsqueeze(1)).float().sum().item()
                current_predictions_cat = torch.cat([i for i in current_predictions], dim=0)
                self.last_day_zero_indexes = classification_preprocessing.update_last_day_zero_indexes(self.last_day_zero_indexes, current_predictions_cat, starting_validation_day)
                starting_validation_day += 1
                val_data = classification_preprocessing.create_val_data(self.prices_dataset, self.prices_dataset, self.encoded_categories, self.preprocessed_time_categories, self.snap_categories_numerical, starting_validation_day, self.last_day_zero_indexes, self.is_more_zeros_than_threshold_list)
                val_dataset = ClassifierValDatset(val_data, self.params['device'])
                self.val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=self.params['batch_size'])
                prediction_values.append(current_predictions_cat)
            predictions = torch.cat([j for j in prediction_values], dim=1)
            file_name = "CLASSIFICATION_Epoch_%d.csv" % self.epochs
            np.savetxt(file_name, predictions.cpu(), delimiter=',', fmt="%s")
            loss_holdout = loss_holdout / total_values
            accuracy = correct_values / total_values
        return float(loss_holdout.detach().cpu().item()), accuracy

