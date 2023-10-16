import random

def shuffle_all_train_choices(train_data, train_labels, seed):
    random.seed(seed)
    shuffled_train_labels = []
    for data, label in zip(train_data, train_labels):
        true_choice = data[label.index(1)]
        random.shuffle(data)
        shuffled_train_labels.append([int(d == true_choice) for d in data])
    return train_data, shuffled_train_labels
