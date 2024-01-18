import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


def activate(z):
    return 1 / (1 + np.exp(-z))


def classify():
    return True


def calculate_logit(weights, bias, attributes):
    return bias + np.dot(attributes, weights)


def softmax(logits):
    exp_logits = np.exp(logits)
    sum_exp_logits = np.sum(exp_logits, axis=-1, keepdims=True)
    probabilities = exp_logits / sum_exp_logits
    return probabilities


def update_weights_and_biases(learning_rate,
                              probabilities,
                              attributes,
                              true_markers,
                              old_weights,
                              old_biases):
    new_weights = np.zeros((len(true_markers), len(attributes)))
    new_biases = np.zeros((len(true_markers)))
    for object_class_index, true_marker in enumerate(true_markers):
        for attribute_index, attribute in enumerate(attributes):
            new_weights[object_class_index][attribute_index] = (old_weights[object_class_index][attribute_index]
                                                                - learning_rate * (probabilities[object_class_index] -
                                                                                   true_markers[object_class_index]) *
                                                                attributes[attribute_index])

        new_biases[object_class_index] = old_biases[object_class_index] - learning_rate * (
                probabilities[object_class_index] -
                true_markers[object_class_index])

    return new_weights, new_biases


def calculate_probabilities(weights, biases, object_classes, attributes):
    object_logits = []
    for object_class_index in object_classes:
        logit = calculate_logit(weights[object_class_index], biases[object_class_index], attributes)
        object_logits.append(logit)

    return softmax(object_logits)


def initiate_weights_and_biases(attributes_count, object_classes_count):
    weights = [[0 for _ in range(attributes_count)] for _ in range(object_classes_count)]
    biases = [0 for _ in range(object_classes_count)]
    return weights, biases


def get_true_markers(object_classes, correct_class_index):
    true_markers = [0 for _ in object_classes]
    true_markers[correct_class_index] = 1
    return true_markers


def validate_accuracy(testing_dataset, object_classes, weights, biases):
    correct_count = 0

    for index, row in testing_dataset.iterrows():
        attributes = list(row[:4])
        target_class_index = int(row[4])

        probabilities = calculate_probabilities(weights, biases, object_classes, attributes)
        result_class_index = np.argmax(probabilities)

        if result_class_index == target_class_index:
            correct_count = correct_count + 1

    return correct_count / testing_dataset.shape[0]


def train_with_set(training_dataset, object_classes, learning_rate, weights, biases):
    for index, row in training_dataset.iterrows():
        attributes = list(row[:4])
        target_class_index = int(row[4])

        probabilities = calculate_probabilities(weights, biases, object_classes, attributes)
        true_markers = get_true_markers(object_classes, target_class_index)
        weights, biases = update_weights_and_biases(learning_rate, probabilities, attributes, true_markers, weights,
                                                    biases)

    return weights, biases


def check_termination_criteria(accuracy_history, convergence_threshold):
    if len(accuracy_history) >= 10:
        std_deviation = np.std(accuracy_history[-10:])
        if std_deviation < convergence_threshold:
            print(f"Converged: Standard Deviation = {std_deviation}")
            return True
        else:
            return False

    return False


def plot_accuracy_graph(accuracy_history, experiment_code):
    fig, ax = plt.subplots()
    ax.plot(accuracy_history)

    ax.set(xlabel='Epoch', ylabel='Accuracy')
    ax.grid()

    plt.savefig(f"{experiment_code}_accuracy_graph.png")


def train(learning_rate):
    object_classes = [0, 1, 2]
    number_of_attributes = 4
    convergence_threshold = 0.001
    weights, biases = initiate_weights_and_biases(number_of_attributes, len(object_classes))

    timestamp = time.time()
    experiment_code = f"{timestamp}_{learning_rate}"

    df = pd.read_csv("Iris.csv")

    df = df.sample(frac=1)
    df.drop(["Id"], axis=1, inplace=True)
    df["Species"] = df["Species"].map({
        "Iris-setosa": 0,
        "Iris-versicolor": 1,
        "Iris-virginica": 2
    }).astype("int64")

    training_dataset = df.sample(frac=0.7, random_state=200)
    testing_dataset = df.drop(training_dataset.index)

    epoch = 0
    accuracy_history = []

    while True:
        weights, biases = train_with_set(training_dataset, object_classes, learning_rate, weights, biases)
        accuracy = validate_accuracy(testing_dataset, object_classes, weights, biases)
        accuracy_history.append(accuracy)
        epoch = epoch + 1
        print(f"Epoch: {epoch}: Accuracy: {accuracy}")
        if check_termination_criteria(accuracy_history, convergence_threshold):
            break

    plot_accuracy_graph(accuracy_history, experiment_code)

    f = open(f"{experiment_code}_stats.txt", "w")
    f.write(f"Epoch: {epoch}, Convergence Accuracy: {accuracy}, Max Accuracy: {max(accuracy_history)}")
    f.close()

    np.savetxt(f"{experiment_code}_weights.txt", weights, fmt='%.6f')
    np.savetxt(f"{experiment_code}_biases.txt", biases, fmt='%.6f')


if __name__ == '__main__':
    for learning_rate in np.arange(0.01, 0.06, 0.01):
        train(learning_rate)
