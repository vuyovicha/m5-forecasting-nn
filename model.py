import torch
import torch.nn as nn
from DilatedRNN import DRNN
import embedding_vectors_preparation
import numpy as np


class ESRNN(nn.Module):
    def __init__(self, train_dataset_len, categories, params):
        super(ESRNN, self).__init__()

        # kind of an alpha and gamma parameters
        create_alpha_level = []
        create_gamma_seasonality = []
        create_seasonality = []

        # PUT ALL CONFIGS INTO ONE FILE
        self.seasonality_parameter = 7  # why so? WHAT VALUE SHOULD BE HERE? 7 is seasonal period for monthly data
        self.output_window_length = 28  # == prediction_horizon
        self.input_window_length = params['input_window_length'] # rule of thumb? todo: getting error on this
        self.LSTM_size = params['LSTM_size']  # I don't know what value should be here

        # smoothing parameters
        for i in range(train_dataset_len):
            create_alpha_level.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            create_gamma_seasonality.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            create_seasonality.append(nn.Parameter(torch.ones(self.seasonality_parameter) * 0.5, requires_grad=True))  # these are initial seasonality values

        self.create_alpha_level = nn.ParameterList(create_alpha_level)
        self.create_gamma_seasonality = nn.ParameterList(create_gamma_seasonality)
        self.create_seasonality = nn.ParameterList(create_seasonality)

        self.sigmoid = nn.Sigmoid()
        self.linear_layer = nn.Linear(self.LSTM_size, self.LSTM_size)  # sizes of input and output sizes respectively
        self.tanh_activation_layer = nn.Tanh()
        self.scoring = nn.Linear(self.LSTM_size, self.output_window_length)  # have no idea what this is for


        self.categories_unique_headers = []
        for j in range(len(categories[0])):
            self.categories_unique_headers.append(embedding_vectors_preparation.create_category_unique_headers(categories, j))  # append the list of unique values of each category

        self.categories_embeddings = []
        for i in range(len(categories[0])):
            self.categories_embeddings.append(embedding_vectors_preparation.create_category_embeddings(self.categories_unique_headers[i]))

        self.all_categories = categories

        # todo set the range of embedding parameters

        self.residual_drnn = ResidualDRNN(self)

        """"
        self.item_id_category_unique_headers = embedding_vectors_preparation.create_category_unique_headers(categories[:, 1])
        self.item_id_category_embedding = embedding_vectors_preparation.create_category_embeddings(self.item_id_category_unique_headers)
        self.dept_id_category_unique_headers = embedding_vectors_preparation.create_category_unique_headers(categories[:, 2])
        self.dept_id_category_embedding = embedding_vectors_preparation.create_category_embeddings(self.dept_id_category_unique_headers)
        self.cat_id_category_unique_headers = embedding_vectors_preparation.create_category_unique_headers(categories[:, 3])
        self.cat_id_category_embedding = embedding_vectors_preparation.create_category_embeddings(self.cat_id_category_embedding)
        self.store_id_category_unique_headers = embedding_vectors_preparation.create_category_unique_headers(categories[:, 4])
        self.store_id_category_embeddings = embedding_vectors_preparation.create_category_embeddings(self.store_id_category_unique_headers)
        self.state_id_category_unique_headers = embedding_vectors_preparation.create_category_unique_headers(categories[:, 5])
        self.state_id_category_embeddings = embedding_vectors_preparation.create_category_embeddings(self.state_id_category_unique_headers)
        """

    def forward(self, train_dataset, val_dataset, indexes, categories):
        train_dataset = train_dataset.float()

        alpha_level = self.sigmoid(torch.stack([self.create_alpha_level[i] for i in indexes]).squeeze(1))
        gamma_seasonality = self.sigmoid(torch.stack([self.create_gamma_seasonality[i] for i in indexes]).squeeze(1))
        initial_seasonality_values = torch.stack([self.create_seasonality[i] for i in indexes])
        seasonalities = []
        for i in range(self.seasonality_parameter):  # unclear totally, it's INITIAL seasonality!!
            seasonalities.append(torch.exp(initial_seasonality_values[:, i]))
        seasonalities.append(torch.exp(initial_seasonality_values[:, 0]))

        levels = []
        difference_of_levels_log = []
        levels.append(train_dataset[:, 0] / seasonalities[0])  # why?
        for i in range(1, train_dataset.shape[1]):
            current_level = alpha_level * (train_dataset[:, i] / seasonalities[i]) + (1 - alpha_level) * levels[i - 1]
            levels.append(current_level)
            difference_of_levels_log.append(torch.log(current_level / levels[i - 1]))
            seasonalities.append(gamma_seasonality * (train_dataset[:, i] / current_level) + (1 - gamma_seasonality) * seasonalities[i])
        stacked_seasonalities = torch.stack(seasonalities).transpose(1, 0)
        stacked_levels = torch.stack(levels).transpose(1, 0)
        # mean_square_error = torch.mean(torch.stack([difference_of_levels_log[i] - difference_of_levels_log[i - 1] ** 2 for i in range(1, len(difference_of_levels_log))]))
        seasonality_extension_begin = stacked_seasonalities.shape[1] - self.seasonality_parameter
        seasonality_extension_end = seasonality_extension_begin - self.seasonality_parameter + self.output_window_length
        stacked_seasonalities = torch.cat((stacked_seasonalities, stacked_seasonalities[:, seasonality_extension_begin:seasonality_extension_end]), dim=1)

        # do a lookup here, we can compute what embedding to add for each TS only once, I think
        input_categories = []
        for i in indexes:
            current_series_categories = []
            for j in range(len(self.categories_unique_headers)):
                #current_category_index = embedding_vectors_preparation.get_category_index(self.categories_unique_headers[j], categories[i][j])
                current_category_index = embedding_vectors_preparation.get_category_index(self.categories_unique_headers[j], self.all_categories[i][j])
                current_series_categories.extend(self.categories_embeddings[j][current_category_index])  # extend, not append
            #print(len(current_series_categories))
            input_categories.append(current_series_categories)
        input_categories = torch.tensor(input_categories)


        input_windows = []
        output_windows = []
        for i in range(self.input_window_length - 1, train_dataset.shape[1]):
            input_window_end = i + 1
            input_window_begin = input_window_end - self.input_window_length
            deseasonalized_input_window = train_dataset[:, input_window_begin:input_window_end] / stacked_seasonalities[:, input_window_begin:input_window_end]
            normalized_input_window = deseasonalized_input_window / stacked_levels[:, i].unsqueeze(1)
            categorized_input_window = torch.cat((normalized_input_window, input_categories), dim=1)
            input_windows.append(categorized_input_window)

            output_window_begin = i + 1
            output_window_end = output_window_begin + self.output_window_length
            if i < train_dataset.shape[1] - self.output_window_length:
                deseasonalized_output_window = train_dataset[:, output_window_begin:output_window_end] / stacked_seasonalities[:, output_window_begin:output_window_end]
                normalized_output_window = deseasonalized_output_window / stacked_levels[:, i].unsqueeze(1)
                output_windows.append(normalized_output_window)
        window_input = torch.cat([i.unsqueeze(0) for i in input_windows], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in output_windows], dim=0)

        self.train()  # tell everyone that training starts
        prediction_values = self.forward_rnn(window_input[:-self.output_window_length])
        actual_values = window_output  # compare network result with actual values, not predicting future here?

        self.eval()  # testing is here?
        holdout_output = self.forward_rnn(window_input)
        holdout_output_reseasonalized = holdout_output[-1] * stacked_seasonalities[:, -self.output_window_length:]
        holdout_output_renormalized = holdout_output_reseasonalized * stacked_levels[:, -1].unsqueeze(1)
        holdout_prediction = holdout_output_renormalized * torch.gt(holdout_output_renormalized, 0).float()
        holdout_actual_values = val_dataset  # there was a test dataset too in the legacy
        holdout_actual_values_deseasonalized = holdout_actual_values.float() / stacked_seasonalities[:, -self.output_window_length:]
        holdout_actual_values_deseasonalized_normalized = holdout_actual_values_deseasonalized / stacked_levels[:, -1].unsqueeze(1)
        self.train()

        return prediction_values, actual_values, holdout_prediction, holdout_output, holdout_actual_values, holdout_actual_values_deseasonalized_normalized

    def forward_rnn(self, dataset):
        dataset = self.residual_drnn(dataset)
        dataset = self.linear_layer(dataset)
        dataset = self.tanh_activation_layer(dataset)
        dataset = self.scoring(dataset)
        return dataset


class ResidualDRNN(nn.Module):
    def __init__(self, ESRNN):
        super(ResidualDRNN, self).__init__()
        layers = []
        dilations = ((1, 2), (2, 6))  # what is the len of this thing? maybe [1, 2, 2,6] or something
        total_embedding_dimensions = embedding_vectors_preparation.get_total_dimensions(ESRNN.categories_unique_headers)  # todo don't forget to uncomment
        input_length = ESRNN.input_window_length + total_embedding_dimensions
        for i in range(len(dilations)):
            layer = DRNN(input_length, ESRNN.LSTM_size, len(dilations[i]), dilations[i], cell_type='LSTM')
            layers.append(layer)
            input_length = ESRNN.LSTM_size

        self.stacked_layers = nn.Sequential(*layers)

    def forward(self, dataset):  # very cunningly, check
        for layer_index in range(len(self.stacked_layers)):
            current_dataset = dataset
            output, _ = self.stacked_layers[layer_index](dataset)  # argument in brackets!
            if layer_index != 0:
                output += current_dataset
            dataset = output

        return output









