import torch
import torch.nn as nn
import embedding_vectors_preparation
import torch.nn.functional as F


class ZeroClassifier(nn.Module):
    def __init__(self, numerical_size, preprocessed_time_categories):
        super(ZeroClassifier, self).__init__()

        unique_labels_length = []
        for j in range(len(preprocessed_time_categories[0])):
            unique_labels_length.append(len(embedding_vectors_preparation.create_category_unique_headers(preprocessed_time_categories, j)))

        embedding_vectors_dimensions = []
        for i in range(len(unique_labels_length)):
            embedding_vectors_dimensions.append(embedding_vectors_preparation.get_embedding_vector_dimension(unique_labels_length[i]))

        self.embeddings = nn.ModuleList([nn.Embedding(unique_labels_length[i], embedding_vectors_dimensions[i]) for i in range(len(unique_labels_length))])
        input_size = embedding_vectors_preparation.get_total_dimensions_from_length(unique_labels_length) + numerical_size

        self.linear_1 = nn.Linear(input_size, 64)
        self.batch_norm_1 = nn.BatchNorm1d(numerical_size)

        self.linear_2 = nn.Linear(64, 16)
        self.batch_norm_2 = nn.BatchNorm1d(64)

        self.linear_3 = nn.Linear(16, 1)
        self.batch_norm_3 = nn.BatchNorm1d(16)

        self.embedding_dropout = nn.Dropout(0.6)
        self.dropouts = nn.Dropout(0.3)

        self.sigmoid_activation = nn.Sigmoid()

    def forward(self, train_dataset_numerical, train_dataset_categorical, validation=False):
        if validation:
            self.eval()
        else:
            self.train()
        output = [embedding(train_dataset_categorical[:, i]) for i, embedding in enumerate(self.embeddings)]
        output = torch.cat(output, dim=1)
        output = self.embedding_dropout(output)
        numerical_input = self.batch_norm_1(train_dataset_numerical)
        output = torch.cat([output, numerical_input], dim=1)
        output = F.relu(self.linear_1(output))
        output = self.dropouts(output)
        output = self.batch_norm_2(output)
        output = F.relu(self.linear_2(output))
        output = self.dropouts(output)
        output = self.batch_norm_3(output)
        output = self.linear_3(output)
        return output


