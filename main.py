import NeuralNetwrk as AI
import numpy as np
import random


class Sample:

    def __init__(self, information, answers):
        self.information = information
        self.answers = answers


if __name__ == '__main__':
    file = open('data.txt', 'r')
    array = []

    for line in file:
        split = (line[:-1].split(','))
        inputs = np.array(split[:3])
        inputs = inputs.astype(np.float64)
        s = 0
        for i in inputs:
            s += i
        inputs /= s

        answer = 0
        if split[-1] == 'setosa':
            answer = [1, 0, 0]
        elif split[-1] == 'versicolor':
            answer = [0, 1, 0]
        elif split[-1] == 'virginica':
            answer = [0, 0, 1]
        array.append(Sample(inputs, answer))

    test_set = []
    test_set[:9] = array[0:9]
    test_set[10:19] = array[10:19]
    test_set[20:] = array[20:29]
    learn_set = []
    learn_set[:39] = array[10:49]
    learn_set[40:79] = array[59:99]
    learn_set[80:] = array[109:]

    random.shuffle(test_set)
    random.shuffle(learn_set)

    network = AI.NeuralNetwork.generate_network([len(array[0].information), 2, len(array[0].answers)])
    network.present()
    for i in range(30):
        for sample in learn_set:
            network.teach_layers(sample.information, sample.answers)

    nice = 0
    network.present()

    for sample in test_set:
        responses = (network.get_responses(sample.information))
        answers = sample.answers
        maxi = 0
        for i in range(len(responses)):
            if responses[i] > responses[maxi]:
                maxi = i
        maxii = 0
        for i in range(len(answers)):
            if answers[i] > answers[maxi]:
                maxii = i
        if maxi == maxii:
            nice += 1

    print(nice)

