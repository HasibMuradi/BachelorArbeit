import math
import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor, random
import numpy as np
import matplotlib.pyplot as plt

import random

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


def converge_success(samples, search_params, epoch):

    gausslist = Main()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    execution = []
    execution.append(0)
    for epo in range(epoch):
        start_time = time.time()
        for i in range(len(samples)):
            likelihood = -torch.log(gausslist(samples[i]))
            likelihood.backward(retain_graph=True)
        end_time = time.time()
        execution.append((end_time - start_time) + execution[epo])

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas.grad)
        optimizer.step()
        optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, execution

class Main(Module):
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


def converge_success_padding(samples, search_params, epoch, batch_size):

    gausslist = Main_padding()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01/len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    # Preprocessing for padding
    batch = []
    index = []
    for n in range(int(len(samples)/batch_size)):
        max_sample_length = max([len(sample) for sample in samples[batch_size * n: batch_size * n + batch_size]])
        batch_matrix = []
        index_matrix = []
        for i in range(batch_size * n, batch_size * n + batch_size):
            batch_matrix.append(samples[i] + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
            index_matrix.append([torch.tensor(1)] * len(samples[i]) + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
        batch.append(batch_matrix)
        index.append(index_matrix)

    guesses = []
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    execution = []
    execution.append(0)
    for epo in range(epoch):
        for i in range(len(index)):
            start_time = time.time()
            likelihood = -torch.log(gausslist(tensor(index[i]), tensor(batch[i])))
            end_time = time.time()
            execution.append((end_time - start_time) + execution[epo*len(index) + i])
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, execution

class Main_padding(Module):
    def forward(self, index_matrix, batch_matrix):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        return torch.prod((((1.0 - l_1_cond) * (density_IRNormal((batch_matrix - self.thetas[2]) / self.thetas[1]) / self.thetas[1])) * index_matrix) + (1-index_matrix),
                          dim=1) * l_1_cond

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()

def exp_performance():
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(0)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for i in range(500):
        samples.append(gausslist.generate())

    guesses1, execution1 = converge_success(samples, search_params, 10, 100)
    guesses2, execution2 = converge_success_padding(samples, search_params, 10, 100)
    plt.plot(execution1, guesses1[:, 0], color="blue")
    plt.plot(execution2, guesses2[:, 0], color="orange")
    plt.axhline(0.8, color="gray", linestyle="dashed")

    plt.xlabel("Training time (s)")
    plt.ylabel(r"Learning parameter ($\theta_0$)")
    plt.legend()
    plt.show()


def exp_convergence():
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(0)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for i in range(500):
        samples.append(gausslist.generate())

    guesses1, execution1 = converge_success(samples, search_params, 10, 100)
    guesses2, execution2 = converge_success_padding(samples, search_params, 10, 100)
    plt.plot(execution1, guesses1[:, 0], color="blue")
    plt.plot(execution2, guesses2[:, 0], color="violet")
    plt.axhline(0.8, color="gray", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel(r"Learning parameter ($\theta_0$)")
    plt.legend()
    #plt.savefig("padding_convergence.png")
    plt.show()


exp_performance()
#exp_convergence()

