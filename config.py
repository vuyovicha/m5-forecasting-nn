import torch


params_init_val = {
            'amount_of_epochs': int(8),
            'learning_rate': 1e-3,
            'optimization_step_size': 5,
            'gamma_coefficient': 0.5,
            'training_percentile': int(45),
            'clip_value': int(23),
            'LSTM_size': int(30),
            'dilations': ((1, 7), (14, 28)),
            'input_window_length': int(28),
            'batch_size': int(6),
            'validation_size': int(28),
            'device': ("cuda" if torch.cuda.is_available() else "cpu"),
            'output_window_length': int(28),
            'training_without_val_dataset': False,
            'starting_validation_day': int(1885),
            'classification_batch_size': int(50)
}

bounds = {
            'amount_of_epochs': (10, 20),  # todo set the bounds
            'learning_rate': (1e-10, 1e-1),
            'optimization_step_size': (0, 10),
            'gamma_coefficient': (0, 1),
            'training_percentile': (0, 100),
            'clip_value': (10, 50),
            'LSTM_size': (10, 50),
            #'dilations': (1, 10),
            'input_window_length': (10, 50)
}