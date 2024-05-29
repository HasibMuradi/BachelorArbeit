import time

import torch
import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor
import numpy as np
import matplotlib.pyplot as plt

import unittest

def rand():
    return np.random.rand()


def randn():
    return np.random.randn()


def hcat(head, tail):
    return [head] + tail


def density_IRNormal(x):
    return (1 / torch.sqrt(torch.tensor(2 * 3.14159))) * torch.exp(-0.5 * x * x)


def contains(elem, list):
    return elem in list


def randomchoice(vector):
    return np.random.choice(np.arange(len(vector)), p=vector / sum(vector))


def create_params():
    return [np.random.rand() * 0.6 + 0.2, np.random.rand() + 0.2, np.random.rand()]

def converge_success(i):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_sequential()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()
    likelihoods = []
    for epoch in range(1):
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            likelihoods.append(likelihood.item())

    return likelihoods

class Main_sequential(Module):
    def forward(self, sample):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        return ((l_1_cond * (1.0 if (sample == []) else 0.0))
                + ((1.0 - l_1_cond)
                   * (0.0 if (sample == []) else
                      ((density_IRNormal((((sample)[0] - self.thetas[2]) / self.thetas[1])) / self.thetas[1])
                       * self.forward((sample)[1:])))))

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()

def converge_success_minibatch(i):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_minibatch()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()
    likelihoods = []
    for epoch in range(1):
        for n in range(10):
            likelihood = -torch.log(gausslist(samples[n*50: n*50+50]))
            likelihoods += likelihood.tolist()

    return likelihoods

class Main_minibatch(Module):
    def forward(self, batch):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        # Define the length of longest sample
        max_sample_length = max([len(sample) for sample in batch])
        # Pad the samples to match the length of the longest sample
        batch_matrix = []
        index_matrix = []
        for i in range(len(batch)):
            batch_matrix.append(batch[i] + [torch.tensor(0)] * max(0, max_sample_length - len(batch[i])))
            index_matrix.append([torch.tensor(1)] * len(batch[i]) + [torch.tensor(0)] * max(0, max_sample_length - len(batch[i])))
        # Convert lists to tensors
        batch_matrix = torch.tensor(batch_matrix)
        index_matrix = torch.tensor(index_matrix)

        # Calculate batch_matrix elements
        batch_matrix = (((1.0 - l_1_cond) * (density_IRNormal((batch_matrix - self.thetas[2]) / self.thetas[1]) / self.thetas[1]))
                         * index_matrix) + (1-index_matrix)

        # Calculate product along dim=1
        return torch.prod(batch_matrix, dim=1) * l_1_cond

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()


def exp_compare_LLHs_of_sequential_and_minibatch():
    sequential_LLHs = converge_success(0)
    minibatch_LLHs = converge_success_minibatch(0)
    #print("seq: ", sequential_LLHs)
    #print("min: ", minibatch_LLHs)
    plt.plot(sequential_LLHs, minibatch_LLHs, marker='o', linestyle='-')
    plt.xlabel('Likelihoods of Sequential Execution')
    plt.ylabel('Likelihoods of Minibatch Execution')
    plt.title('Likelihoods Comparison of Sequential and Minibatch Execution')
    plt.grid(True)
    plt.savefig('exp_LLHs_comparison.png')
    plt.show()

#exp_compare_LLHs_of_sequential_and_minibatch()


class TestLLHCompare(unittest.TestCase):

    def test_CompareLLHs(self):
        sequential_LLHs = converge_success(0)
        minibatch_LLHs = converge_success_minibatch(0)

        tolerance_percent = 0.00001
        for i in range(500):
            absolute_difference = abs(sequential_LLHs[i] - minibatch_LLHs[i])
            tolerance_value = (tolerance_percent/100 * max(abs(sequential_LLHs[i]), abs(minibatch_LLHs[i])))
            assert absolute_difference <= tolerance_value, \
                f"Values are not within {tolerance_percent}% tolerance. " \
                f"Absolute difference: {absolute_difference}, Tolerance: {tolerance_value}"

if __name__ == '__main__':
    unittest.main()
