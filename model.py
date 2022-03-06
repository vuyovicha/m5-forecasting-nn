import torch
import torch.nn as nn
from DilatedRNN import DRNN
import embedding_vectors_preparation
import numpy as np


class ESRNN(nn.Module):
    def __init__(self, train_dataset_len, categories, time_categories, params, used_days_dataset, predictions_indexes, predictions_lengths, zero_related_predictions_indexes, real_values_starting_indexes):
        super(ESRNN, self).__init__()
        self.params = params
        self.used_days_dataset = used_days_dataset

        # kind of an alpha and gamma parameters
        create_alpha_level = []
        create_gamma_seasonality = []
        create_seasonality = []

        self.seasonality_parameter = 7  # why so? WHAT VALUE SHOULD BE HERE? 7 is seasonal period for monthly data
        self.output_window_length = params['output_window_length']  # == prediction_horizon
        self.input_window_length = params['input_window_length']  # rule of thumb?
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
        self.scoring = nn.Linear(self.LSTM_size, 1)

        self.categories_unique_headers = []
        for j in range(len(categories[0])):
            self.categories_unique_headers.append(embedding_vectors_preparation.create_category_unique_headers(categories, j))  # append the list of unique values of each category

        categories_embeddings = []
        self.categories_starting_indexes = []
        for i in range(len(categories[0])):
            current_amount_of_unique_values = len(self.categories_unique_headers[i])
            current_embedding_vector_dimension = embedding_vectors_preparation.get_embedding_vector_dimension(current_amount_of_unique_values)
            current_embedding_list_for_tensor = [0.001 for j in range(current_embedding_vector_dimension)]
            for j in range(current_amount_of_unique_values):
                categories_embeddings.append(nn.Parameter(torch.Tensor(current_embedding_list_for_tensor), requires_grad=True))
                if j == 0:
                    self.categories_starting_indexes.append(len(categories_embeddings) - 1)
        self.categories_embeddings = nn.ParameterList(categories_embeddings)
        self.all_categories = categories

        self.time_categories_unique_headers = []
        for j in range(len(time_categories[0])):
            self.time_categories_unique_headers.append(embedding_vectors_preparation.create_category_unique_headers(time_categories, j))

        time_categories_embeddings = []
        self.time_categories_starting_indexes = []
        for i in range(len(time_categories[0])):
            current_amount_of_unique_values = len(self.time_categories_unique_headers[i])
            current_embedding_vector_dimension = embedding_vectors_preparation.get_embedding_vector_dimension(current_amount_of_unique_values)
            current_embedding_list_for_tensor = [0.001 for j in range(current_embedding_vector_dimension)]
            for j in range(current_amount_of_unique_values):
                time_categories_embeddings.append(nn.Parameter(torch.Tensor(current_embedding_list_for_tensor), requires_grad=True))
                if j == 0:
                    self.time_categories_starting_indexes.append(len(time_categories_embeddings) - 1)
        self.time_categories_embeddings = nn.ParameterList(time_categories_embeddings)
        self.all_time_categories = time_categories

        embedding_dimensions = embedding_vectors_preparation.get_total_dimensions(self.categories_unique_headers)
        embedding_dimensions += embedding_vectors_preparation.get_total_dimensions(self.time_categories_unique_headers)
        self.total_embedding_dimensions = embedding_dimensions

        self.residual_drnn = ResidualDRNN(self)

        self.predictions_indexes = predictions_indexes
        self.predictions_lengths = predictions_lengths
        self.zero_related_predictions_indexes = zero_related_predictions_indexes
        self.real_values_starting_indexes = real_values_starting_indexes

    def forward(self, train_dataset, val_dataset, indexes, categories, validation=False, training_without_val_dataset=False):
        train_dataset = train_dataset.float()

        categories_embeddings = []
        for i in range(len(self.categories_embeddings)):
            categories_embeddings.append(self.tanh_activation_layer(self.categories_embeddings[i]))

        input_categories_list = []
        for j in indexes:
            current_series_categories = []
            for k in range(len(self.categories_unique_headers)):
                current_category_index = embedding_vectors_preparation.get_category_index(self.categories_unique_headers[k], self.all_categories[j][k])
                current_series_categories.append(categories_embeddings[current_category_index + self.categories_starting_indexes[k]])
            input_categories_list.append(torch.cat([i.unsqueeze(0) for i in current_series_categories], dim=1).squeeze())

        time_categories_embeddings = []
        for i in range(len(self.time_categories_embeddings)):
            time_categories_embeddings.append(self.tanh_activation_layer(self.time_categories_embeddings[i]))

        input_time_categories_list = []
        for j in range(len(self.all_time_categories)):
            current_day_categories = []
            for k in range(len(self.time_categories_unique_headers)):
                current_day_category_index = embedding_vectors_preparation.get_category_index(self.time_categories_unique_headers[k], self.all_time_categories[j][k])
                current_day_categories.append(time_categories_embeddings[current_day_category_index + self.time_categories_starting_indexes[k]])
            input_time_categories_list.append(torch.cat([i.unsqueeze(0) for i in current_day_categories], dim=1).squeeze())
        input_time_categories = torch.cat([i.unsqueeze(0) for i in input_time_categories_list], dim=0)

        alpha_level = self.sigmoid(torch.stack([self.create_alpha_level[i] for i in indexes]).squeeze(1))
        gamma_seasonality = self.sigmoid(torch.stack([self.create_gamma_seasonality[i] for i in indexes]).squeeze(1))
        initial_seasonality_values = torch.stack([self.create_seasonality[i] for i in indexes])

        series_seasonalities = []
        for i in range(len(initial_seasonality_values)):
            per_series_initial_seasonality = []
            for j in range(self.seasonality_parameter):
                per_series_initial_seasonality.append(initial_seasonality_values[i, j])
            per_series_initial_seasonality.append(initial_seasonality_values[i, 0])
            series_seasonalities.append(per_series_initial_seasonality)

        series_levels = []
        for i in range(len(train_dataset)):
            seasonality_index = 0
            per_series_levels = []
            for j in range(train_dataset.shape[1]):
                if train_dataset[i, j] != 0:
                    if seasonality_index == 0:
                        per_series_levels.append(train_dataset[i, j] / series_seasonalities[i][0])
                    else:
                        per_series_levels.append(alpha_level[i] * (train_dataset[i, j] / series_seasonalities[i][seasonality_index]) + (1 - alpha_level[i]) * per_series_levels[-1])
                        series_seasonalities[i].append(gamma_seasonality[i] * (train_dataset[i, j] / per_series_levels[-1]) + (1 - gamma_seasonality[i]) * series_seasonalities[i][seasonality_index])
                    seasonality_index += 1
            series_levels.append(per_series_levels)

        for i in range(len(train_dataset)):
            seasonality_extension_begin = len(series_seasonalities[i]) - self.seasonality_parameter
            seasonality_extension_end = seasonality_extension_begin - self.seasonality_parameter + self.predictions_lengths[indexes[i]]
            series_seasonalities[i].extend(series_seasonalities[i][seasonality_extension_begin:seasonality_extension_end])

        cat_series_seasonalities_list = []
        for i in range(len(series_seasonalities)):
            cat_series_seasonalities_list.append(torch.stack(series_seasonalities[i]))

        cat_series_levels_list = []
        for i in range(len(series_levels)):
            cat_series_levels_list.append(torch.stack(series_levels[i]))

        input_values_per_series = []
        output_values_per_series = []
        for i in range(len(train_dataset)):
            current_input_values = []
            current_output_values = []
            current_non_zero_value_index = 0
            for j in range(train_dataset.shape[1]):
                if train_dataset[i, j] == 0:
                    current_input_values.append(torch.zeros(1 + self.total_embedding_dimensions).to(self.params['device']))
                else:
                    deseasonalized_input_value = train_dataset[i, j] / cat_series_seasonalities_list[i][current_non_zero_value_index]
                    normalized_input_value = deseasonalized_input_value / cat_series_levels_list[i][current_non_zero_value_index]
                    categorized_input_value = torch.cat((normalized_input_value.unsqueeze(0), input_categories_list[i]), dim=0)
                    input_time_category_index = self.used_days_dataset[indexes[i]][current_non_zero_value_index]
                    if j != train_dataset.shape[1] - 1:
                        time_categorized_input_value = torch.cat((categorized_input_value, input_time_categories[input_time_category_index + 1]), dim=0)
                    elif self.predictions_lengths[indexes[i]] != 0:
                        first_validation_index = self.predictions_indexes[indexes[i]][0]
                        time_categorized_input_value = torch.cat((categorized_input_value, input_time_categories[first_validation_index]), dim=0)
                    else:
                        time_categorized_input_value = torch.cat((categorized_input_value, input_time_categories[0]), dim=0)  # inputing random category, not gonna use this value
                    current_input_values.append(time_categorized_input_value)
                    current_non_zero_value_index += 1
                if j < train_dataset.shape[1] - 1:
                    if train_dataset[i, j] != 0:
                        deseasonalized_output_value = train_dataset[i, j + 1] / cat_series_seasonalities_list[i][current_non_zero_value_index]
                        normalized_output_value = deseasonalized_output_value / cat_series_levels_list[i][current_non_zero_value_index - 1]
                        current_output_values.append(normalized_output_value)
                    else:
                        current_output_values.append(train_dataset[i, j + 1])
            input_values_per_series.append(current_input_values)
            output_values_per_series.append(current_output_values)

        input_values_per_series_list = []
        output_values_per_series_list = []
        for i in range(len(input_values_per_series[0])):
            input_values_per_series_list.append(torch.cat([series_inputs[i].unsqueeze(0) for series_inputs in input_values_per_series], dim=0))
            if i < len(output_values_per_series[0]):
                output_values_per_series_list.append(torch.stack([series_outputs[i].unsqueeze(0) for series_outputs in output_values_per_series], dim=0))

        cat_input_values = torch.cat([i.unsqueeze(0) for i in input_values_per_series_list], dim=0)
        cat_output_values = torch.cat([i.unsqueeze(0) for i in output_values_per_series_list], dim=0)

        self.train()
        prediction_values = self.forward_rnn(cat_input_values[:-1])
        actual_values = cat_output_values

        if validation:
            self.eval()
            all_holdout_outputs = []
            for i in range(len(indexes)):
                current_holdout_outputs = []
                current_input = self.create_current_input(cat_input_values, i)
                holdout_output = self.forward_rnn(current_input)
                current_holdout_outputs.append(holdout_output[0, -1])
                for j in range(self.predictions_lengths[indexes[i]] - 1):
                    categorized_input_value = torch.cat((current_holdout_outputs[-1], input_categories_list[i]), dim=0)
                    time_categorized_input_value = torch.cat((categorized_input_value, input_time_categories[self.predictions_indexes[indexes[i]][j + 1]]), dim=0)
                    current_input = torch.cat((current_input, time_categorized_input_value.unsqueeze(0).unsqueeze(0)), dim=0)
                    holdout_output = self.forward_rnn(current_input)
                    current_holdout_outputs.append(holdout_output[0, -1])
                all_holdout_outputs.append(current_holdout_outputs)

            holdout_outputs_list = []
            for i in range(len(all_holdout_outputs)):
                holdout_outputs_list.append(torch.cat([j for j in all_holdout_outputs[i]]))

            dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
            renormalized_holdout_outputs_list = []
            for i in range(len(holdout_outputs_list)):
                holdout_output_reseasonalized = holdout_outputs_list[i] * cat_series_seasonalities_list[i][-self.predictions_lengths[indexes[i]]:]
                holdout_output_renormalized = holdout_output_reseasonalized * cat_series_levels_list[i][-1]  # todo not correct level index here
                holdout_output_zero_compared = holdout_output_renormalized * torch.gt(holdout_output_renormalized, 0).float()
                holdout_output_insert_ones_list = [torch.ones(1).type(dtype)[0] if value < 1 else value for value in holdout_output_zero_compared]  # ADDED THIS BECAUSE THESE VALUES ARE NOT ZERO
                holdout_output_insert_ones = torch.cat([j.unsqueeze(0) for j in holdout_output_insert_ones_list])
                renormalized_holdout_outputs_list.append(holdout_output_insert_ones)

            holdout_outputs_zero_list = []
            for i in range(len(renormalized_holdout_outputs_list)):
                current_holdout_outputs_zero_list = [0 for k in range(self.output_window_length)]
                for j in range(len(self.zero_related_predictions_indexes[indexes[i]])):
                    current_holdout_outputs_zero_list[self.zero_related_predictions_indexes[indexes[i]][j]] = renormalized_holdout_outputs_list[i][j].cpu().numpy()
                current_holdout_outputs_zero_numpy_list = np.array(current_holdout_outputs_zero_list)
                holdout_outputs_zero_list.append(torch.from_numpy(current_holdout_outputs_zero_numpy_list).type(dtype))

            holdout_prediction = torch.cat([i.unsqueeze(0) for i in holdout_outputs_zero_list], dim=0)
            holdout_actual_values = val_dataset

            if training_without_val_dataset:
                return holdout_prediction

            real_output_values = []
            for i in range(len(train_dataset)):
                real_output_values.append(torch.cat((torch.zeros(self.real_values_starting_indexes[indexes[i]]).to(self.params['device']), train_dataset[i, self.real_values_starting_indexes[indexes[i]]:]), dim=0))
            cat_real_output_values = torch.cat([i.unsqueeze(0) for i in real_output_values], dim=0)

            normalized_model_output_list = []
            for series_index in range(prediction_values.shape[1]):
                current_normalized_model_output_list = []
                for value_index in range(len(self.used_days_dataset[indexes[series_index]]) - 1):
                    reseasonalized_value = prediction_values[self.used_days_dataset[indexes[series_index]][value_index], series_index] * cat_series_seasonalities_list[series_index][value_index]
                    renormalized_value = reseasonalized_value * cat_series_levels_list[series_index][value_index]  # maybe smth weird happens with indexes here
                    current_normalized_model_output_list.append(renormalized_value * torch.gt(renormalized_value, 0).float())
                real_values = torch.cat([i for i in current_normalized_model_output_list])
                zeros = torch.zeros(self.real_values_starting_indexes[indexes[series_index]] + 1).to(self.params['device'])
                normalized_model_output_list.append(torch.cat((zeros, real_values), dim=0))
            cat_normalized_model_output_list = torch.cat([i.unsqueeze(0) for i in normalized_model_output_list])

            self.train()

            return prediction_values, actual_values, holdout_prediction, holdout_actual_values, cat_real_output_values, cat_normalized_model_output_list

        else:
            return prediction_values, actual_values

    def forward_rnn(self, dataset):
        dataset = self.residual_drnn(dataset)
        dataset = self.linear_layer(dataset)
        dataset = self.tanh_activation_layer(dataset)
        dataset = self.scoring(dataset)
        return dataset

    def create_current_input(self, cat_input_values, index):  # triple []
        current_input_list = []
        for j in range(len(cat_input_values)):
            temp_list = []
            temp_list.append(cat_input_values[j, index])
            current_input_list.append(torch.cat([i.unsqueeze(0) for i in temp_list]))
        cat_current_input_list = torch.cat([j.unsqueeze(0) for j in current_input_list], dim=0)
        return cat_current_input_list


class ResidualDRNN(nn.Module):
    def __init__(self, ESRNN):
        super(ResidualDRNN, self).__init__()
        layers = []
        dilations = ESRNN.params['dilations']  # TODO has been changed according to ESRNN daily config

        input_length = 1 + ESRNN.total_embedding_dimensions

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
