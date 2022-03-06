import torch
import torch.nn as nn
import numpy as np


def get_embedding_vector_dimension(amount_of_unique_values):
    dimension = min((amount_of_unique_values + 1) // 2, 50)
    if dimension == 0:
        return 1
    else:
        return dimension


def create_category_unique_headers(whole_dataset_categories, j):
    whole_dataset_extracted_categories = extract_needed_category(whole_dataset_categories, j)
    category_unique_values_numpy = np.unique(whole_dataset_extracted_categories)
    category_unique_values = []
    for i in range(len(category_unique_values_numpy)):
        category_unique_values.append(category_unique_values_numpy[i])
    return category_unique_values


def extract_needed_category(whole_dataset_categories, j):
    extracted_category = []
    for i in range(len(whole_dataset_categories)):
        extracted_category.append(whole_dataset_categories[i][j])
    return np.array(extracted_category)


def category_not_added(dataset_category_value, category_unique_values):
    for i in range(len(category_unique_values)):
        temp_category_unique_values = str(category_unique_values[i])
        temp_dataset_category_value = str(dataset_category_value)
        if temp_category_unique_values == temp_dataset_category_value:
            return False
    return True


def create_category_embeddings(category_unique_headers):  # maybe the problem will occur - try to do this in the model init
    amount_of_unique_values = len(category_unique_headers)
    embedding_vector_dimension = get_embedding_vector_dimension(amount_of_unique_values)
    return [nn.Parameter(torch.ones(embedding_vector_dimension) * 0.001, requires_grad=True) for i in range(amount_of_unique_values)]
    # each unique value within one category gets torch.tensor of specified length - that means list of lists of tensors


def get_total_dimensions(categories_unique_headers):
    sum_of_dimensions = 0
    for i in range(len(categories_unique_headers)):
        sum_of_dimensions += get_embedding_vector_dimension(len(categories_unique_headers[i]))
    return sum_of_dimensions


def get_total_dimensions_from_length(unique_labels_length):
    sum_of_dimensions = 0
    for i in range(len(unique_labels_length)):
        sum_of_dimensions += get_embedding_vector_dimension(unique_labels_length[i])
    return sum_of_dimensions


def get_category_index(category_unique_headers, category):
    for i in range(len(category_unique_headers)):
        if category_unique_headers[i] == category:
            return i
    return False
