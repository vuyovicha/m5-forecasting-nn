import torch
import torch.nn as nn
from DilatedRNN import DRNN

class ESRNN(nn.Module):
    def __init__(self, train_dataset_len):
        super(ESRNN, self).__init__()
        self.train_dataset_len = train_dataset_len

        # kind of an alpha and gamma parameters
        create_alpha_level = []
        create_betta_seasonality = []

        create_seasonality = []

        # PUT ALL CONFIGS INTO ONE FILE
        self.seasonality_parameter = 7.0  # why so? WHAT VALUE SHOULD BE HERE?
        self.output_window_length = 28  # == prediction_horizon
        self.input_window_length = 15  # rule of thumb?
        self.LSTM_size = 30  # I don't know what value should be here

        # smoothing parameters
        for i in range(train_dataset_len):
            create_alpha_level.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            create_betta_seasonality.append(nn.Parameter(torch.Tensor([0.5]), requires_grad=True))
            create_seasonality.append(nn.Parameter(torch.ones(self.seasonality_parameter * 0.5), requires_grad=True))

        self.create_alpha_level = nn.ParameterList(create_alpha_level)
        self.create_betta_seasonality = nn.ParameterList(create_betta_seasonality)
        self.create_seasonality = nn.ParameterList(create_seasonality)

        self.sigmoid = nn.Sigmoid()
        self.linear_layer = nn.Linear(self.LSTM_size, self.LSTM_size)  # sizes of input and output sizes respectively
        self.tanh_activation_layer = nn.Tanh()
        self.scoring = nn.Linear(self.LSTM_size, self.output_window_length)  # have no idea what this is for

        self.residual_drnn = ResidualDRNN()

    def forward(self, train_dataset, val_dataset, indexes):
        alpha_level = self.sigmoid(torch.stack([self.create_alpha_level[i] for i in indexes]).squeeze(1))  # level smoothing
        betta_seasonality = self.sigmoid(torch.stack([self.create_betta_seasonality[i] for i in indexes]).squeeze(1))  # seasonality smoothing
        temp_for_seasonality = torch.stack([self.create_seasonality[i] for i in indexes])

        seasonalities = []

        for i in range (self.seasonality_parameter):  # unclear totally
            seasonalities.append(torch.exp(temp_for_seasonality[:, i]))
        seasonalities.append(torch.exp(temp_for_seasonality[:, 0]))

        train_dataset = train_dataset.float()

        levels = []
        difference_of_levels_log = []

        levels.append(train_dataset[:, 0] / seasonalities[0])  # why?
        for i in range(1, train_dataset.shape[1]):
            current_level = alpha_level * (train_dataset[:, i] / seasonalities[i]) + (1 - alpha_level) * levels[i - 1]
            levels.append(current_level)
            difference_of_levels_log.append(torch.log(current_level / levels[i - 1]))
            seasonalities.append(betta_seasonality * (train_dataset[:, i] / current_level) + (1 - betta_seasonality) * seasonalities[i])

        stacked_seasonalities = torch.stack(seasonalities).transpose(1, 0)
        stacked_levels = torch.stack(levels).transpose(1, 0)

        mean_square_error = torch.mean(torch.stack([difference_of_levels_log[i] - difference_of_levels_log[i - 1] ** 2 for i in range(1, len(difference_of_levels_log))]))

        seasonality_extension_begin = stacked_seasonalities.shape[1] - self.seasonality_parameter
        seasonality_extension_end = seasonality_extension_begin - self.seasonality_parameter + self.output_window_length
        stacked_seasonalities = torch.cat((stacked_seasonalities, stacked_seasonalities[:, seasonality_extension_begin:seasonality_extension_end]), dim=1)

        input_windows = []
        output_windows = []
        for i in range(self.input_window_length - 1, train_dataset.shape[1]):
            input_window_end = i + 1
            input_window_begin = input_window_end - self.input_window_length
            deseasonalized_input_window = train_dataset[:, input_window_begin:input_window_end] / stacked_seasonalities[:, input_window_begin:input_window_end]
            normalized_input_window = deseasonalized_input_window / stacked_levels[:, i].unsqueeze(1)
            # category should be here?
            input_windows.append(normalized_input_window)

            output_window_begin = i + 1
            output_window_end = output_window_begin + self.output_window_length
            if i < train_dataset.shape[1] - self.output_window_length:
                deseasonalized_output_window = train_dataset[:, output_window_begin:output_window_end] / stacked_seasonalities[:, output_window_begin:output_window_end]
                normalized_output_window = deseasonalized_output_window / stacked_levels[:, i].unsqueeze(1)
                output_windows.append(normalized_input_window)

        window_input = torch.cat([i.unsqueeze(0) for i in input_windows], dim=0)
        window_output = torch.cat([i.unsqueeze(0) for i in output_windows], dim=0)

        self.train()  # tell everyone that training starts

        prediction_values = self.forward_rnn(window_input[:-self.output_window_length])
        actual_values = window_output

        self.eval()  # testing is here?
        holdout_output = self.forward_rnn(window_input)
        holdout_output_reseasonalized = holdout_output[-1] * stacked_seasonalities[:, -self.output_window_length]
        holdout_output_renormalized = holdout_output_reseasonalized * stacked_levels[:, -1].unsqueeze(1)
        holdout_prediction = holdout_output_renormalized * torch.gt(holdout_output_renormalized, 0).float()
        holdout_actual_values = val_dataset  # there was a test dataset too in the legacy
        holdout_actual_values_deseasonalized = holdout_actual_values.float() / stacked_seasonalities[:, -self.output_window_length]
        holdout_actual_values_deseasonalized_normalized = holdout_actual_values_deseasonalized / stacked_levels[:, -1].unsqueeze(1)

        self.train()

        return prediction_values, actual_values, holdout_prediction, holdout_output, holdout_actual_values, holdout_actual_values_deseasonalized_normalized, mean_square_error

    def forward_rnn(self, dataset):
        dataset = self.residual_drnn(dataset)
        dataset = self.linear_layer(dataset)
        dataset = self.tanh_activation_layer(dataset)
        dataset = self.scoring(dataset)
        return dataset


class ResidualDRNN(nn.Module):
    def __init__(self):
        super(ResidualDRNN, self).__init__()

        layers = []
        amount_of_categories = 6  # not this number actually, see provided data?
        dilations = ((1, 2), (2, 6))  #what is the len of this thing?

        input_length = ESRNN.input_window_length + amount_of_categories
        for i in range(len(dilations)):
            layer = DRNN(input_length, ESRNN.LSTM_size, len(dilations[i]), dilations[i], 'GRU')
            layers.append(layer)
            input_length = ESRNN.LSTM_size

        self.stacked_layers = nn.Sequential(*layers)

    def forward(self, dataset):  # very cunningly, check
        for layer_index in range(len(self.stacked_layers)):
            current_dataset = dataset
            output = self.stacked_layers[layer_index](dataset)  # not sure what brackets mean here
            if layer_index != 0:
                output += current_dataset
            dataset = output

        return output









