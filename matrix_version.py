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

def converge_success(i):
    #variant one: initialize seed only here, set to i.
    #variant two: set to 1 here, set to i thereafter.
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_sequential()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))
    print("\tSample Params: ", gausslist.thetas)

    samples = []
    for n in range(500):
        x = gausslist.generate()
        # print(x)
        # print(gausslist.forward(x))
        # print()
        # if x != []:
        samples.append(x)
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    print("\t", gausslist.thetas)
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()
    guesses = []
    for epoch in range(10):
        likelihoods = 0
        undiffs = 0
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            if type(likelihood) is float:
                undiffs += 1
                likelihoods += likelihood
            else:
                likelihoods += likelihood.item()
                likelihood.backward(retain_graph=True)
        print("iteration report: ", epoch)
        print("\taggregate likelihood = {}".format(likelihoods / len(samples)))
        print("\t", gausslist.thetas.grad)
        optimizer.step()
        optimizer.zero_grad()
        print("\t{} / {} samples are undiff".format(undiffs, len(samples)))
        print("\t", gausslist.thetas)
        guesses.append([gausslist.thetas[0].item() ,gausslist.thetas[1].item() ,gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return (guesses, sample_params)

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
    #variant one: initialize seed only here, set to i.
    #variant two: set to 1 here, set to i thereafter.
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_minibatch()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))
    print("\tSample Params: ", gausslist.thetas)

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    print("\t", gausslist.thetas)
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()
    guesses = []
    n = 0
    #print("sam: ", samples)
    #print("ran: ", random.sample(samples, 10))
    for epoch in range(10):
        likelihoods = 0
        for n in range(int(len(samples)/100)):
            likelihood = -torch.log(gausslist(samples[n*100: n*100+100]))
            likelihood = torch.sum(likelihood)
            likelihoods += likelihood.item()
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
        """if ((n+1)*100+100) <= len(samples):
            n += 1
        else:
            n = 0"""
        print("iteration report: ", epoch)
        print("\taggregate likelihood = {}".format(likelihoods / len(samples)))
        print("\t", gausslist.thetas.grad)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params

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


def exp_convergence_success():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for i in range(10):
        c = next(color)
        guesses, sample_params = converge_success(i)
        tgt = sample_params[0]
        graph = guesses[:, 0]
        bad = False
        graph = graph - np.ones_like(guesses[:, 0]) * tgt
        if graph[0] > 0 and graph[-1] < 0:
            bad = True
        if graph[0] < 0 and graph[-1] > 0:
            bad = True
        if abs(graph[0]) < abs(graph[-1]):
            bad = True
        bad = True
        if bad:
            plt.plot(guesses[:, 0], color=c, label="recurser{}".format(i))
            plt.plot(np.ones_like(guesses[:, 0]) * sample_params[0], color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('matrix_version_sequential_convergence.png')
    plt.show()

def exp_converge_success_minibatch():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for i in range(10):
        c = next(color)
        guesses, sample_params = converge_success_minibatch(i)
        tgt = sample_params[0]
        graph = guesses[:, 0]
        bad = False
        graph = graph - np.ones_like(guesses[:, 0]) * tgt
        if graph[0] > 0 and graph[-1] < 0:
            bad = True
        if graph[0] < 0 and graph[-1] > 0:
            bad = True
        if abs(graph[0]) < abs(graph[-1]):
            bad = True
        bad = True
        if bad:
            plt.plot(guesses[:, 0], color=c, label="recurser{}".format(i))
            plt.plot(np.ones_like(guesses[:, 0]) * sample_params[0], color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('matrix_version_minibatch_convergence.png')
    plt.show()

def exp_performance_minibatch():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    recurser = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    c = next(color)
    execution_times1 = []
    for i in range(10):
        start_time1 = time.time()
        guesses, sample_params = converge_success(i)
        end_time1 = time.time()
        execution_time1 = end_time1 - start_time1
        execution_times1.append(execution_time1)

    c = next(color)
    execution_times2 = []
    for i in range(10):
        start_time2 = time.time()
        guesses, sample_params = converge_success_minibatch(i)
        end_time2 = time.time()
        execution_time2 = end_time2 - start_time2
        execution_times2.append(execution_time2)

    mean1 = np.mean(execution_times1)
    mean2 = np.mean(execution_times2)
    x = round(mean1 / mean2, 2)
    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x))
    plt.scatter(recurser, execution_times1, marker='o', label="Sequential execution")
    plt.scatter(recurser, execution_times2, marker='+', label="Minibatch execution")
    plt.xlabel("Recurser")
    plt.ylabel("Execution time (s)")
    plt.legend()
    plt.savefig('matrix_version_performance.png')
    plt.show()

def exp_converge_sequential_minibatch():

    guesses, sample_params = converge_success(0)
    guesses1, sample_params1 = converge_success_minibatch(0)
    print("gue1: ", guesses[:, 0])
    print("gue2: ", guesses1[:, 0])

    plt.plot(guesses[:, 0], color="blue", label="BGD")
    plt.plot(np.linspace(0, 9, 50), guesses1[:, 0], color="green", label="MGD")
    #plt.plot((np.ones_like(guesses[:, 0]) * sample_params[0]-guesses[:, 0])+np.ones_like(guesses[:, 0]) * sample_params[0], color=c)
    plt.plot(np.ones_like(guesses[:, 0]) * sample_params[0], color="gray", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.savefig('exp_convergence_minibatch.png')
    plt.show()



#exp_convergence_success()
#exp_converge_success_minibatch()
#exp_performance_minibatch()
exp_converge_sequential_minibatch()