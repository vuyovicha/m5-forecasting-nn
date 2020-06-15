import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import classification_preprocessing
from dataset import ClassifierDataset, ClassifierValDatset
from classification_model import ZeroClassifier


class GlobalClassificationTrainer(nn.Module):
    def __init__(self, train_dataset, prices_dataset, preprocessed_time_categories, snap_categories_numerical, val_targets, params):
        super(GlobalClassificationTrainer, self).__init__()
        self.train_dataset = train_dataset
        self.prices_dataset = prices_dataset
        self.preprocessed_time_categories = preprocessed_time_categories
        self.snap_categories_numerical = snap_categories_numerical
        self.val_targets = val_targets
        self.params = params
        self.starting_validation_day = params['starting_validation_day']

    def train_series_classifiers(self):
        predictions = []
        accuracy = 0
        train_loss = 0
        val_loss = 0
        for series in range(len(self.train_dataset)):
            classification_dataset = classification_preprocessing.create_series_classification_dataset(self.train_dataset[series], self.prices_dataset[series], self.preprocessed_time_categories, self.snap_categories_numerical)
            classification_train_dataset = ClassifierDataset(classification_dataset, self.params['device'])
            train_data_loader = DataLoader(classification_train_dataset, shuffle=True, batch_size=self.params['classification_batch_size'])

            last_train_value = 0 if self.train_dataset[series, self.starting_validation_day - 1] == 0 else 1
            initial_val_value = classification_preprocessing.create_val_value(self.prices_dataset[series], self.preprocessed_time_categories, self.snap_categories_numerical, self.starting_validation_day, last_train_value)
            initial_val_dataset = ClassifierValDatset(initial_val_value, self.params['device'])
            initial_val_data_loader = DataLoader(initial_val_dataset, shuffle=False, batch_size=1)

            print("SERIES %d" % series)
            classification_model = ZeroClassifier(len(classification_dataset[0, 0:5]), self.preprocessed_time_categories)
            current_accuracy, current_predictions, current_train_loss, current_val_loss = SeriesClassificationTrainer(classification_model, train_data_loader, initial_val_data_loader, self.params, self.val_targets[series], self.prices_dataset[series], self.preprocessed_time_categories, self.snap_categories_numerical).train_epochs()
            print("OVERALL SERIES")
            print("Accuracy %f" % current_accuracy)
            print("Train loss %f" % current_train_loss)
            print("Val loss %f" % current_val_loss)
            print()
            accuracy += current_accuracy
            train_loss += current_train_loss
            val_loss += current_val_loss
            predictions.append(current_predictions)

        accuracy = float(accuracy) / len(self.train_dataset)
        train_loss = float(train_loss) / len(self.train_dataset)
        val_loss = float(val_loss) / len(self.train_dataset)
        print("TOTAL_ACCURACY: %f" % accuracy)
        print("TOTAL_TRAIN_LOSS: %f" % train_loss)
        print("TOTAL_VAL_LOSS: %f" % val_loss)
        print()
        predictions_cat = torch.stack(predictions)
        file_name = "CLASSIFICATION.csv"
        np.savetxt(file_name, predictions_cat.cpu(), delimiter=',', fmt="%s")


class SeriesClassificationTrainer(nn.Module):
    def __init__(self, model, train_data_loader, initial_val_data_loader, params, series_val_targets, series_prices, preprocessed_time_categories, snap_categories_numerical):
        super(SeriesClassificationTrainer, self).__init__()
        self.model = model.to(params['device'])
        self.train_data_loader = train_data_loader
        self.val_data_loader = initial_val_data_loader

        self.learning_rate = params['learning_rate']
        self.optimization = torch.optim.Adam(self.model.parameters(), self.learning_rate)
        self.amount_of_epochs = params['amount_of_epochs']
        self.epochs = 0

        self.criterion = nn.BCEWithLogitsLoss()
        self.params = params
        self.series_val_targets = series_val_targets
        self.series_prices = series_prices
        self.preprocessed_time_categories = preprocessed_time_categories
        self.snap_categories_numerical = snap_categories_numerical
        self.starting_validation_day = params['starting_validation_day']

    def train_epochs(self):
        max_accuracy = 0
        predictions = []
        predictions.append(torch.zeros(self.params['output_window_length']))
        train_loss = 1e8
        val_loss = 1e8
        for epoch in range(self.amount_of_epochs):
            loss_epoch = self.train_batches()
            if not self.params['training_without_val_dataset']:
                loss_validation, accuracy, current_predictions = self.validation()
                if max_accuracy < accuracy:
                    predictions.append(current_predictions)
                    max_accuracy = accuracy
                    val_loss = loss_validation
                    train_loss = loss_epoch
            else:
                self.training_without_val_set()
        return max_accuracy, predictions[-1], train_loss, val_loss

    def train_batches(self):
        self.model.train()
        loss_epoch = 0
        total_values = 0
        correct_overall = 0
        for batch, (numerical_data, categorical_data, target, index) in enumerate(self.train_data_loader):
            current_loss, current_values, correct_values = self.train_batch(numerical_data, categorical_data, target, index)
            correct_overall += correct_values
            loss_epoch += current_values * current_loss
            total_values += current_values
        self.epochs += 1
        accuracy = float(correct_overall) / total_values
        return float(loss_epoch) / total_values

    def train_batch(self, numerical_data, categorical_data, target, index):
        batch_size = target.shape[0]
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
            validation_day = self.starting_validation_day
            for i in range(self.params['output_window_length']):
                for batch, (numerical_data, categorical_data, index) in enumerate(self.val_data_loader):
                    output = self.model(numerical_data, categorical_data, True)
                    targets = torch.tensor(self.series_val_targets[i])
                    stack_targets = targets.type_as(output)
                    current_holdout_loss = self.criterion(output, stack_targets.unsqueeze(0).unsqueeze(1))
                    loss_holdout += current_holdout_loss
                    total_values += 1
                    rounded_output = torch.round(torch.sigmoid(output))  # added sigmoid here
                    if self.series_prices[validation_day] == 0:
                        rounded_output = 0
                    prediction_values.append(rounded_output)
                    correct_values += (rounded_output == stack_targets).float().sum().item()
                validation_day += 1
                val_value = classification_preprocessing.create_val_value(self.series_prices, self.preprocessed_time_categories, self.snap_categories_numerical, validation_day, rounded_output)
                val_dataset = ClassifierValDatset(val_value, self.params['device'])
                self.val_data_loader = DataLoader(val_dataset, shuffle=False, batch_size=1)
            predictions = torch.cat([j[0] for j in prediction_values])
            loss_holdout = loss_holdout / total_values
            accuracy = correct_values / total_values
        return float(loss_holdout.detach().cpu().item()), accuracy, predictions

