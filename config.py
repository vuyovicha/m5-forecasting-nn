params_init_val = {
            'amount_of_epochs': 16,
            'learning_rate': 1e-3,
            'optimization_step_size': 5,
            'gamma_coefficient': 0.5,
            'training_percentile': int(45),
            'clip_value': int(23),
            'LSTM_size': int(30),
            #'dilations' :((1, 2), (2, 6)),
            'input_window_length': int(28)
}

bounds = {
            'amount_of_epochs': (10, 20), # todo set the bounds
            'learning_rate': (1e-10, 1e-1),
            'optimization_step_size': (0, 10),
            'gamma_coefficient': (0, 1),
            'training_percentile': (0, 100),
            'clip_value': (10, 50),
            'LSTM_size': (10, 50),
            #'dilations': (1, 10),
            'input_window_length': (10, 50)
}