import torch
import torch.nn as nn
from DilatedRNN import DRNN
import embedding_vectors_preparation


class ESRNN(nn.Module):
    def __init__(self, train_dataset_len, categories, params, preprocessed_time_categories):
        super(ESRNN, self).__init__()
        self.params = params

        self.seasonality_parameter = params['seasonality_parameter']
        self.output_window_length = params['output_window_length']
        self.input_window_length = params['input_window_length']
        self.LSTM_size = params['LSTM_size']

        self.sigmoid = nn.Sigmoid()
        self.linear_layer = nn.Linear(self.LSTM_size, self.LSTM_size)  # sizes of input and output sizes respectively
        self.tanh_activation_layer = nn.Tanh()
        self.scoring = nn.Linear(self.LSTM_size, 1)

        # next two loops are like in the classification model - can do that stuff somewhere outside
        unique_labels_length = []
        for j in range(len(preprocessed_time_categories[0])):
            unique_labels_length.append(len(embedding_vectors_preparation.create_category_unique_headers(preprocessed_time_categories, j)))

        embedding_dimensions = 0
        embedding_vectors_dimensions = []
        for i in range(len(unique_labels_length)):
            current_dimension = embedding_vectors_preparation.get_embedding_vector_dimension(unique_labels_length[i])
            embedding_vectors_dimensions.append(current_dimension)
            embedding_dimensions += current_dimension

        self.time_embeddings = nn.ModuleList([nn.Embedding(unique_labels_length[i], embedding_vectors_dimensions[i]) for i in range(len(unique_labels_length))])
        self.encoded_time_categories = preprocessed_time_categories

        unique_labels_length = []
        for j in range(len(categories[0])):
            unique_labels_length.append(len(embedding_vectors_preparation.create_category_unique_headers(preprocessed_time_categories, j)))

        embedding_vectors_dimensions = []
        for i in range(len(unique_labels_length)):
            current_dimension = embedding_vectors_preparation.get_embedding_vector_dimension(unique_labels_length[i])
            embedding_vectors_dimensions.append(embedding_vectors_preparation.get_embedding_vector_dimension(unique_labels_length[i]))
            embedding_dimensions += current_dimension

        self.series_embeddings = nn.ModuleList([nn.Embedding(unique_labels_length[i], embedding_vectors_dimensions[i]) for i in range(len(unique_labels_length))])
        self.encoded_categories = categories
        self.total_embedding_dimensions = embedding_dimensions

        self.residual_drnn = ResidualDRNN(self)

    def forward(self, train_dataset, val_dataset, indexes, categories, validation=False, training_without_val_dataset=False):
        train_dataset = train_dataset.float()

        input_categories_list = []
        for j in indexes:
            input_categories_list.append(torch.cat([embedding(torch.tensor(self.encoded_categories[j][i]).to(self.params['device'])) for i, embedding in enumerate(self.series_embeddings)]))
        input_series_categories = torch.stack(input_categories_list)

        input_time_categories_list = []
        for day in range(len(self.encoded_time_categories)):
            current_embeddings = torch.cat([embedding(torch.tensor(self.encoded_time_categories[day][i]).to(self.params['device'])) for i, embedding in enumerate(self.time_embeddings)])
            input_time_categories_list.append(torch.stack([current_embeddings for series in indexes]))
        input_time_categories = torch.stack(input_time_categories_list)

        input_values = []
        output_values = []
        for day in range(train_dataset.shape[1]):
            time_categorized_value = torch.cat((train_dataset[:, day].unsqueeze(1), input_time_categories[day, :]), dim=1)
            series_categorized_value = torch.cat((time_categorized_value, input_series_categories), dim=1)
            input_values.append(series_categorized_value)

            if day < train_dataset.shape[1] - 1:
                output_values.append(train_dataset[:, day + 1].unsqueeze(1))

        cat_input_values = torch.cat([i.unsqueeze(0) for i in input_values], dim=0)
        cat_output_values = torch.cat([i.unsqueeze(0) for i in output_values], dim=0)

        self.train()
        prediction_values = self.forward_rnn(cat_input_values[:-1])
        actual_values = cat_output_values

        if validation:
            self.eval()
            current_holdout_output = self.forward_rnn(cat_input_values)
            holdout_output = []
            holdout_output.append(current_holdout_output[-1])
            for i in range(self.params['output_window_length'] - 1):
                time_categorized_value = torch.cat((holdout_output[-1], input_time_categories[day, :]), dim=1)
                series_categorized_value = torch.cat((time_categorized_value, input_series_categories), dim=1)
                input_values.append(series_categorized_value)
                cat_input_values = torch.cat([j.unsqueeze(0) for j in input_values], dim=0)
                current_holdout_output = self.forward_rnn(cat_input_values)
                holdout_output.append(current_holdout_output[-1])

            holdout_prediction_stack = torch.stack(holdout_output).transpose(1, 0)
            holdout_output_cat_list = []
            for i in range(len(holdout_prediction_stack)):
                holdout_output_cat_list.append(torch.cat([j for j in holdout_prediction_stack[i]]))
            holdout_output_cat = torch.cat([i.unsqueeze(0) for i in holdout_output_cat_list], dim=0)
            holdout_prediction = holdout_output_cat
            holdout_actual_values = val_dataset

            if training_without_val_dataset:
                return holdout_prediction

            self.train()

            return prediction_values, actual_values, holdout_prediction, holdout_actual_values

        else:
            return prediction_values, actual_values

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
        dilations = ESRNN.params['dilations']
        input_length = 1 + ESRNN.total_embedding_dimensions

        for i in range(len(dilations)):
            layer = DRNN(input_length, ESRNN.LSTM_size, len(dilations[i]), dilations[i], cell_type='LSTM')
            layers.append(layer)
            input_length = ESRNN.LSTM_size

        self.stacked_layers = nn.Sequential(*layers)

    def forward(self, dataset):
        for layer_index in range(len(self.stacked_layers)):
            current_dataset = dataset
            output, _ = self.stacked_layers[layer_index](dataset)
            if layer_index != 0:
                output += current_dataset
            dataset = output

        return output
