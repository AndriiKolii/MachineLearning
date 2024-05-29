import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def fit(data_x, data_y):
    classes = np.unique(data_y)
    mean = {}
    dev = {}
    class_prior = {}

    for i in classes:
        mean[i] = np.mean(data_x[data_y == i], axis=0)
        dev[i] = np.std(data_x[data_y == i], axis=0)
        class_prior[i] = np.mean(data_y == i)

    return mean, dev, class_prior


def predict(x, mean, dev, class_prior):
    post = np.zeros((x.shape[0], len(class_prior)))
    for i, k in enumerate(class_prior):
        prob = (1 / (np.sqrt(2 * np.pi * dev[k]))) * np.exp(-0.5 * ((x - mean[k]) ** 2 / dev[k] ** 2))
        post[:, i] = np.prod(prob, axis=1) * class_prior[k]
    predicted_y = np.argmax(post, axis=1)
    predicted_y = np.where(predicted_y == 0, -1, 1)
    return predicted_y


if __name__ == '__main__':
    data_x = np.array([(5.3, 2.3), (5.7, 2.5), (4.0, 1.0), (5.6, 2.4), (4.5, 1.5), (5.4, 2.3), (4.8, 1.8), (4.5, 1.5), (5.1, 1.5), (6.1, 2.3), (5.1, 1.9), (4.0, 1.2), (5.2, 2.0), (3.9, 1.4), (4.2, 1.2), (4.7, 1.5), (4.8, 1.8), (3.6, 1.3), (4.6, 1.4), (4.5, 1.7), (3.0, 1.1), (4.3, 1.3), (4.5, 1.3), (5.5, 2.1), (3.5, 1.0), (5.6, 2.2), (4.2, 1.5), (5.8, 1.8), (5.5, 1.8), (5.7, 2.3), (6.4, 2.0), (5.0, 1.7), (6.7, 2.0), (4.0, 1.3), (4.4, 1.4), (4.5, 1.5), (5.6, 2.4), (5.8, 1.6), (4.6, 1.3), (4.1, 1.3), (5.1, 2.3), (5.2, 2.3), (5.6, 1.4), (5.1, 1.8), (4.9, 1.5), (6.7, 2.2), (4.4, 1.3), (3.9, 1.1), (6.3, 1.8), (6.0, 1.8), (4.5, 1.6), (6.6, 2.1), (4.1, 1.3), (4.5, 1.5), (6.1, 2.5), (4.1, 1.0), (4.4, 1.2), (5.4, 2.1), (5.0, 1.5), (5.0, 2.0), (4.9, 1.5), (5.9, 2.1), (4.3, 1.3), (4.0, 1.3), (4.9, 2.0), (4.9, 1.8), (4.0, 1.3), (5.5, 1.8), (3.7, 1.0), (6.9, 2.3), (5.7, 2.1), (5.3, 1.9), (4.4, 1.4), (5.6, 1.8), (3.3, 1.0), (4.8, 1.8), (6.0, 2.5), (5.9, 2.3), (4.9, 1.8), (3.3, 1.0), (3.9, 1.2), (5.6, 2.1), (5.8, 2.2), (3.8, 1.1), (3.5, 1.0), (4.5, 1.5), (5.1, 1.9), (4.7, 1.4), (5.1, 1.6), (5.1, 2.0), (4.8, 1.4), (5.0, 1.9), (5.1, 2.4), (4.6, 1.5), (6.1, 1.9), (4.7, 1.6), (4.7, 1.4), (4.7, 1.2), (4.2, 1.3), (4.2, 1.3)])
    data_y = np.array([1, 1, -1, 1, -1, 1, 1, -1, 1, 1, 1, -1, 1, -1, -1, -1, -1, -1, -1, 1, -1, -1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, -1, -1, 1, 1, -1, -1, 1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, -1, 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, 1, -1, 1, 1, 1, -1, 1, -1, 1, 1, 1, 1, -1, -1, 1, 1, -1, -1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, -1])

    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.3)

    mean, std, class_prior = fit(x_train, y_train)
    predicted_y = predict(x_test, mean, std, class_prior)
    accuracy = np.mean(predicted_y == y_test)

    print(f'Mistakes: {len(y_test) - np.sum(predicted_y == y_test)}')
    print(f'Percent of mistakes: {round(100 * (1 - accuracy), 3)}%')

    class1_x = [x_train[i] for i in range(len(x_train)) if y_train[i] == 1]
    class2_x = [x_train[i] for i in range(len(x_train)) if y_train[i] == -1]

    class1_x_values, class1_y_values = zip(*class1_x)
    class2_x_values, class2_y_values = zip(*class2_x)

    plt.figure(figsize=(8, 6))
    plt.scatter(class1_x_values, class1_y_values, color='blue', label='Class 1')
    plt.scatter(class2_x_values, class2_y_values, color='red', label='Class -1')
    plt.title('Train Set')
    plt.legend()
    plt.grid(True)
    plt.show()
