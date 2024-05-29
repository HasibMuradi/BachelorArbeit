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


def converge_success_flattening(i):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_flattening()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for i in range(int(len(samples)/100)):
            likelihood = -torch.log(gausslist(samples[100*i: 100*i+100]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
        end_time = time.time()
        execution.append(end_time - start_time)
        print("iteration report: ", epoch)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params


class Main_flattening(Module):
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

        lengths = []
        lst = []
        for i in range(len(batch)):
            if batch[i]:
                lengths += [i] * len(batch[i])
                lst += batch[i]

        lengths = tensor(lengths)
        lst = tensor(lst)

        return ((torch.ones(len(batch))
                .index_reduce_(0, lengths, ((1.0 - l_1_cond) * (density_IRNormal((lst - self.thetas[2]) / self.thetas[1]) / self.thetas[1])), 'prod'))
                * l_1_cond)


    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()


"""def exp_performance():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    recurser = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    c = next(color)
    execution_times1 = []
    for i in range(10):
        start_time1 = time.time()
        guesses, sample_params = converge_success_matrix(i)
        end_time1 = time.time()
        execution_time1 = end_time1 - start_time1
        execution_times1.append(execution_time1)

    c = next(color)
    execution_times2 = []
    for i in range(10):
        start_time2 = time.time()
        guesses, sample_params = converge_success_flattening(i)
        end_time2 = time.time()
        execution_time2 = end_time2 - start_time2
        execution_times2.append(execution_time2)

    mean1 = np.mean(execution_times1)
    mean2 = np.mean(execution_times2)
    x = round(mean1 / mean2, 2)
    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x))
    plt.scatter(recurser, execution_times1, marker='o', label="Minibatch execution")
    plt.scatter(recurser, execution_times2, marker='+', label="Matrix execution")
    plt.xlabel("Epoch")
    plt.ylabel("Execution time (s)")
    plt.legend()
    # plt.savefig('matrix_version_performance.png')
    plt.show()"""


def exp_convergence():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))

    for i in range(10):
        c = next(color)
        guesses, sample_params = converge_success_flattening(i)
        plt.plot(np.linspace(0, 10, 50), guesses[:, 0], color=c, label="Recurser: {}".format(i))
        plt.axhline(sample_params[0], color="gray", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


# exp_performance_()
exp_convergence()
