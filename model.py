import torch
import torch.nn as nn
from DilatedRNN import DRNN
import embedding_vectors_preparation


class ESRNN(nn.Module):
    def __init__(self, train_dataset_len, categories, time_categories, params, prices_dataset, used_days_dataset):
        super(ESRNN, self).__init__()
        self.params = params
        self.prices_dataset = prices_dataset
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
        #self.scoring = nn.Linear(self.LSTM_size, self.output_window_length)
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
        #input_categories = torch.cat([i.unsqueeze(0) for i in input_categories_list], dim=0)  # squeeze or unsqueeze?

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
            seasonality_extension_end = seasonality_extension_begin - self.seasonality_parameter + self.output_window_length
            series_seasonalities[i].extend(series_seasonalities[i][seasonality_extension_begin:seasonality_extension_end])


        """""
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
        input_categories = torch.cat([i.unsqueeze(0) for i in input_categories_list], dim=0)  # squeeze or unsqueeze?

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
        
        seasonalities = []
        for i in range(self.seasonality_parameter):  # unclear totally, it's INITIAL seasonality!!
            seasonalities.append(torch.exp(initial_seasonality_values[:, i]))
        seasonalities.append(torch.exp(initial_seasonality_values[:, 0]))
        
        # removed difference_of_levels_log from the loop
        levels = []
        levels.append(train_dataset[:, 0] / seasonalities[0])
        for i in range(1, train_dataset.shape[1]):
            current_level = alpha_level * (train_dataset[:, i] / seasonalities[i]) + (1 - alpha_level) * levels[i - 1]
            levels.append(current_level)
            seasonalities.append(gamma_seasonality * (train_dataset[:, i] / current_level) + (1 - gamma_seasonality) * seasonalities[i])

        stacked_seasonalities = torch.stack(seasonalities).transpose(1, 0)
        stacked_levels = torch.stack(levels).transpose(1, 0)

        # we go to the seasonality of the output window here, always looking for m values behind
        seasonality_extension_begin = stacked_seasonalities.shape[1] - self.seasonality_parameter
        seasonality_extension_end = seasonality_extension_begin - self.seasonality_parameter + self.output_window_length  # todo do we need to add output length here? maybe just 1
        stacked_seasonalities = torch.cat((stacked_seasonalities, stacked_seasonalities[:, seasonality_extension_begin:seasonality_extension_end]), dim=1)
        
        input_values = []
        output_values = []
        for i in range(train_dataset.shape[1]):
            deseasonalized_input_value = train_dataset[:, i] / stacked_seasonalities[:, i]
            normalized_input_value = deseasonalized_input_value / stacked_levels[:, i]
            categorized_input_value = torch.cat((normalized_input_value.unsqueeze(1), input_categories), dim=1)
            multiply_input_time_categories = torch.cat([input_time_categories[i + 1].unsqueeze(0) for j in range(len(train_dataset))], dim=0)  # concatenating FOLLOWING time stamp category
            time_categorized_input_value = torch.cat((categorized_input_value, multiply_input_time_categories), dim=1)
            input_values.append(time_categorized_input_value)

            if i < train_dataset.shape[1] - 1:
                deseasonalized_output_value = train_dataset[:, i + 1] / stacked_seasonalities[:, i + 1]
                normalized_output_value = deseasonalized_output_value / stacked_levels[:, i]  # .unsqueeze(1)  # TODO why is here the same index - do we need unsqueeze?
                output_values.append(normalized_output_value.unsqueeze(1))

        cat_input_values = torch.cat([i.unsqueeze(0) for i in input_values], dim=0)
        cat_output_values = torch.cat([i.unsqueeze(0) for i in output_values], dim=0)  # shape 0 == 1884
        """""

        input_values_per_series = []
        output_values_per_series = []
        for i in range(len(train_dataset)):
            current_input_values = []
            current_output_values = []
            current_non_zero_value_index = 0
            for j in range(train_dataset.shape[1]):
                if train_dataset[i, j] == 0:
                    current_input_values.append(torch.zeros(1 + self.total_embedding_dimensions))
                else:
                    deseasonalized_input_value = train_dataset[i, j] / series_seasonalities[i][current_non_zero_value_index]
                    normalized_input_value = deseasonalized_input_value / series_levels[i][current_non_zero_value_index]
                    categorized_input_value = torch.cat((normalized_input_value.unsqueeze(0), input_categories_list[i]), dim=0)
                    input_time_category_index = self.used_days_dataset[indexes[i]][current_non_zero_value_index]
                    time_categorized_input_value = torch.cat((categorized_input_value, input_time_categories[input_time_category_index + 1]), dim=0)
                    current_input_values.append(time_categorized_input_value)
                    current_non_zero_value_index += 1
                if j < train_dataset.shape[1] - 1:
                    if train_dataset[i, j] != 0:
                        deseasonalized_output_value = train_dataset[i, j + 1] / series_seasonalities[i][current_non_zero_value_index]
                        normalized_output_value = deseasonalized_output_value / series_levels[i][current_non_zero_value_index - 1]
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
        prediction_values = self.forward_rnn(cat_input_values[:-1])  # shape 0 == 1884
        actual_values = cat_output_values

        if validation:
            self.eval()
            prediction_timestamps_indexes = []
            current_holdout_output = self.forward_rnn(cat_input_values)
            holdout_output = []
            holdout_output.append(current_holdout_output[-1])
            for i in range(self.output_window_length - 1):
                prediction_timestamps_indexes.append(train_dataset.shape[1] + i)
                multiply_input_time_categories = torch.cat([input_time_categories[train_dataset.shape[1] + i + 1].unsqueeze(0) for j in range(len(train_dataset))], dim=0)
                categorized_input_value = torch.cat((holdout_output[-1], input_categories), dim=1)
                input_values.append(torch.cat((categorized_input_value, multiply_input_time_categories), dim=1))
                cat_input_values = torch.cat([j.unsqueeze(0) for j in input_values], dim=0)
                current_holdout_output = self.forward_rnn(cat_input_values)
                holdout_output.append(current_holdout_output[-1])
            prediction_timestamps_indexes.append(train_dataset.shape[1] + self.output_window_length - 1)
            holdout_output_stack = torch.stack(holdout_output).transpose(1, 0)
            holdout_output_cat_list = []
            for i in range(len(holdout_output_stack)):
                holdout_output_cat_list.append(torch.cat([j for j in holdout_output_stack[i]]))
            holdout_output_cat = torch.cat([i.unsqueeze(0) for i in holdout_output_cat_list], dim=0)

            holdout_output_cat_reseasonalized = holdout_output_cat * stacked_seasonalities[:, -self.output_window_length:]
            holdout_output_cat_renormalized = holdout_output_cat_reseasonalized * stacked_levels[:, -1].unsqueeze(1)
            holdout_prediction = holdout_output_cat_renormalized * torch.gt(holdout_output_cat_renormalized, 0).float()

            """""
            current_timestamp_index = 0
            for i in range(prediction_timestamps_indexes[0], prediction_timestamps_indexes[0] + len(prediction_timestamps_indexes)):
                for j in indexes:
                    if self.prices_dataset[j, i] == 0:
                        holdout_prediction[j % self.params['batch_size'], current_timestamp_index] = 0
                current_timestamp_index += 1
            """""

            if training_without_val_dataset:
                return holdout_prediction

            holdout_actual_values = val_dataset
            holdout_actual_values_deseasonalized = holdout_actual_values.float() / stacked_seasonalities[:, -self.output_window_length:]
            holdout_actual_values_deseasonalized_normalized = holdout_actual_values_deseasonalized / stacked_levels[:, -1].unsqueeze(1)
            self.train()

            return prediction_values, actual_values, holdout_prediction, holdout_output_cat, holdout_actual_values, holdout_actual_values_deseasonalized_normalized

        else:
            return prediction_values, actual_values


        """""
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
        input_categories = torch.cat([i.unsqueeze(0) for i in input_categories_list], dim=0)  # squeeze or unsqueeze?

        time_categories_embeddings = []
        for i in range(len(self.time_categories_embeddings)):
            time_categories_embeddings.append(self.tanh_activation_layer(self.time_categories_embeddings[i]))

        input_time_categories_list = []
        for j in range(train_dataset.shape[1]):
            current_day_categories = []
            for k in range(len(self.time_categories_unique_headers)):
                current_day_category_index = embedding_vectors_preparation.get_category_index(self.time_categories_unique_headers[k], self.all_time_categories[j][k])
                current_day_categories.append(time_categories_embeddings[current_day_category_index + self.time_categories_starting_indexes[k]])
            input_time_categories_list.append(torch.cat([i.unsqueeze(0) for i in current_day_categories], dim=1).squeeze())
        input_time_categories = torch.cat([i.unsqueeze(0) for i in input_time_categories_list], dim=0)
        
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
        seasonality_extension_begin = stacked_seasonalities.shape[1] - self.seasonality_parameter
        seasonality_extension_end = seasonality_extension_begin - self.seasonality_parameter + self.output_window_length
        stacked_seasonalities = torch.cat((stacked_seasonalities, stacked_seasonalities[:, seasonality_extension_begin:seasonality_extension_end]), dim=1)
        
        input_values = []
        for i in range(train_dataset.shape[1]):
            deseasonalized_value = train_dataset[:, i] / stacked_seasonalities[:, i]
            normalized_value = deseasonalized_value / stacked_levels[:, i]
            multiply_input_time_categories = torch.cat([input_time_categories[i].unsqueeze(0) for j in range(len(train_dataset))], dim=0)
            time_categorized_value = torch.cat((normalized_value.unsqueeze(1), multiply_input_time_categories), dim=1)
            input_values.append(time_categorized_value)

        input_windows = []
        output_windows = []
        for i in range(self.input_window_length - 1, train_dataset.shape[1]):
            input_window_index_end = i + 1
            input_window_index_begin = input_window_index_end - self.input_window_length
            input_cat_values = torch.cat([input_values[j] for j in range(input_window_index_begin, input_window_index_end)], dim=1)
            categorized_input_window = torch.cat((input_cat_values, input_categories), dim=1)
            input_windows.append(categorized_input_window)

            output_window_begin = i + 1
            output_window_end = output_window_begin + self.output_window_length
            if i < train_dataset.shape[1] - self.output_window_length:
                deseasonalized_output_window = train_dataset[:, output_window_begin:output_window_end] / stacked_seasonalities[:, output_window_begin:output_window_end]
                normalized_output_window = deseasonalized_output_window / stacked_levels[:, i].unsqueeze(1)
                output_windows.append(normalized_output_window)


        # the first version: 28 days windowing and whole TS embeddings
        #input_windows = []
        #output_windows = []
        #for i in range(self.input_window_length - 1, train_dataset.shape[1]):
            #input_window_end = i + 1
            #input_window_begin = input_window_end - self.input_window_length
            #deseasonalized_input_window = train_dataset[:, input_window_begin:input_window_end] / stacked_seasonalities[:, input_window_begin:input_window_end]
            #normalized_input_window = deseasonalized_input_window / stacked_levels[:, i].unsqueeze(1)
            #categorized_input_window = torch.cat((normalized_input_window, input_categories), dim=1)
            #input_windows.append(categorized_input_window)

            #output_window_begin = i + 1
            #output_window_end = output_window_begin + self.output_window_length
            #if i < train_dataset.shape[1] - self.output_window_length:
                #deseasonalized_output_window = train_dataset[:, output_window_begin:output_window_end] / stacked_seasonalities[:, output_window_begin:output_window_end]
                #normalized_output_window = deseasonalized_output_window / stacked_levels[:, i].unsqueeze(1)
                #output_windows.append(normalized_output_window)
                


        window_input = torch.cat([i.unsqueeze(0) for i in input_windows], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in output_windows], dim=0)

        self.train()  # tell everyone that training starts
        prediction_values = self.forward_rnn(window_input[:-self.output_window_length])
        actual_values = window_output  # compare network result with actual values, not predicting future here?

        self.eval()  # testing is here?
        holdout_output = self.forward_rnn(window_input)
        #print(holdout_output[-1])
        holdout_output_reseasonalized = holdout_output[-1] * stacked_seasonalities[:, -self.output_window_length:]
        holdout_output_renormalized = holdout_output_reseasonalized * stacked_levels[:, -1].unsqueeze(1)
        holdout_prediction = holdout_output_renormalized * torch.gt(holdout_output_renormalized, 0).float()
        holdout_actual_values = val_dataset  # there was a test dataset too in the legacy
        holdout_actual_values_deseasonalized = holdout_actual_values.float() / stacked_seasonalities[:, -self.output_window_length:]
        holdout_actual_values_deseasonalized_normalized = holdout_actual_values_deseasonalized / stacked_levels[:, -1].unsqueeze(1)
        self.train()

        return prediction_values, actual_values, holdout_prediction, holdout_output, holdout_actual_values, holdout_actual_values_deseasonalized_normalized
        """""

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
        dilations = ESRNN.params['dilations']  # TODO has been changed according to ESRNN daily config

        #total_embedding_dimensions = embedding_vectors_preparation.get_total_dimensions(ESRNN.categories_unique_headers)
        #total_embedding_dimensions += ESRNN.input_window_length * embedding_vectors_preparation.get_total_dimensions(ESRNN.time_categories_unique_headers)
        #total_embedding_dimensions += embedding_vectors_preparation.get_total_dimensions(ESRNN.time_categories_unique_headers)
        #input_length = ESRNN.input_window_length + total_embedding_dimensions  # todo add time embedding dimesnions
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









