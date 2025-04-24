import json
import os
import random
from sklearn.model_selection import train_test_split

def load_instructions_from_json(dataset_path: str, label_list: list[str], train_size: float=1.0):
    """
    Load data from a JSON file and split the dataset into training and testing sets based on the provided labels.
    
    Args:
        dataset_path (str): The path to the dataset (JSON file).
        label_list (list[str]): A list of labels to extract from the dataset.
        train_size (float): The proportion of data to use for the training set (default is 1.0, meaning all data is used for training).
    
    Returns:
        dict: A dictionary containing the dataset name, label list, and the training and testing splits.
        list: A list of entities (e.g., city names).
    """
    assert 0 < train_size <= 1.0, "train_size should be in (0, 1]"
    
    # Read the JSON dataset file
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    # Initialize a dictionary to hold the return data structure
    ret = {
        "dataset_name": os.path.basename(dataset_path),
        "label_list": label_list,
        "train": [],
        "test": [],  # Placeholder for training and testing data
    }

    # Dictionary to hold label-specific data
    label_data = {label: [] for label in label_list}
    entities = []  # List to store entities (e.g., city names)
    
    # Loop through the dataset and extract text samples for each label
    for item in dataset:
        entity = item["ent"]  # Get the entity name (e.g., city)
        entities.append(entity)
        for label in label_list:
            if label in item:
                texts = item[label]  # Get texts associated with this label
                for text in texts:
                    label_data[label].append((text, label, entity))  # Save the text, label, and entity

    # Split the data for each label into training and testing sets
    for label in label_list:
        label_samples = label_data[label]  # Get all samples for this label
        
        # Shuffle the samples to randomize their order
        random.shuffle(label_samples)
        
        # Split into training and testing sets based on train_size
        train_data, test_data = train_test_split(label_samples, test_size=1 - train_size, random_state=42)
        
        # Append the split data to the results
        ret["train"].append(train_data)
        ret["test"].append(test_data)

    # Print the size of the training and testing sets
    train_size = sum(len(train) for train in ret["train"])
    test_size = sum(len(test) for test in ret["test"])
    
    print(f"Total training samples: {train_size}, Total testing samples: {test_size}")
    print(f"Training to testing ratio: {train_size / (train_size + test_size):.2f} : {(test_size / (train_size + test_size)):.2f}")

    return ret, entities  # Return the dataset split and the list of entities


def load_instructions_by_size(
    dataset_path: str,
    label_list: list[str],
    train_size: float = 1.0,
):
    """
    Load data with a specific training set size and split it into training and testing sets.
    
    Args:
        dataset_path (str): The path to the dataset (JSON file).
        label_list (list[str]): A list of labels to extract from the dataset.
        train_size (float): The proportion of data to use for the training set (default is 1.0).
    
    Returns:
        dict: A dictionary containing the dataset name, label list, and the training and testing splits.
        list: A list of entities (e.g., city names).
    """
    return load_instructions_from_json(dataset_path, label_list, train_size)


def load_instructions_by_flag(
    dataset_path: str,
    label_list: list[str],
):
    """
    Load data and split it based on predefined training and testing flags within the dataset.
    
    Args:
        dataset_path (str): The path to the dataset (JSON file).
        label_list (list[str]): A list of labels to extract from the dataset.
    
    Returns:
        dict: A dictionary containing the dataset name, label list, and the training and testing splits.
        list: A list of entities (e.g., city names).
    """
    return load_instructions_from_json(dataset_path, label_list)
