import torch
import torch.nn as nn
import numpy as np


def get_embedding_vector_dimension(amount_of_unique_values):
    return min(amount_of_unique_values // 2, 50)


def create_category_unique_headers(whole_dataset_categories):
    category_unique_values = np.swapaxes(whole_dataset_categories, 0, 1)
    for i in range (len(whole_dataset_categories)):
        if not (whole_dataset_categories[i] in category_unique_values):
            category_unique_values.append(whole_dataset_categories[i])
    return np.array(category_unique_values)


def create_category_embeddings(category_unique_headers):  # maybe the problem will occur - try to do this in the model init
    amount_of_unique_values = len(category_unique_headers)
    embedding_vector_dimension = get_embedding_vector_dimension(amount_of_unique_values)
    return [nn.Parameter(torch.ones(embedding_vector_dimension) * 0.001, requires_grad=True) for i in amount_of_unique_values]


def get_total_dimensions(categories_unique_headers):
    sum_of_dimensions = 0
    for i in range(len(categories_unique_headers)):
        sum_of_dimensions += categories_unique_headers[i].shape[1]
    return sum_of_dimensions
