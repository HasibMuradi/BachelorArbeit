import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor
import numpy as np
import matplotlib.pyplot as plt
import math

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
    #for j in range(10):
        #try:
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main1()

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
        #except Exception:
            #print("Failed to converge, iteration: {}".format(i))
class Main1(Module):
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


def batch_converge_success(i):
    #variant one: initialize seed only here, set to i.
    #variant two: set to 1 here, set to i thereafter.
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)

    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    batches_of_samples = []
    for n in range(500):
        x = gausslist.generate()
        flag = False
        index = 0
        for batch in batches_of_samples:
            if (len(x) < len(batch[0])):
                break
            else:
                if (len(x) == len(batch[0])):
                    batch.append(x)
                    flag = True
            index += 1
        if (flag == False):
            batches_of_samples.insert(index, [x])

    samples_length = sum(len(batch) for batch in batches_of_samples)
    """for batch in batches_of_samples:
        print("sam: ", len(batch), len(batch[0]))
    print("batches: ", len(batches_of_samples))"""
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / samples_length, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()
    guesses = []
    for epoch in range(10):
        likelihoods = 0
        for batch in batches_of_samples:
            likelihood = -torch.log(gausslist(batch))
            likelihood = torch.sum(likelihood)
            likelihoods += likelihood.item()
            likelihood.backward(retain_graph=True)
        print("iteration report:", epoch)
        print("\taggregate likelihood = {}".format(likelihoods / samples_length))
        print("\t", gausslist.thetas.grad)
        optimizer.step()
        optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item() ,gausslist.thetas[1].item() ,gausslist.thetas[2].item()])

    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return (guesses, sample_params)

class Main(Module):
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

        return l_1_cond * torch.prod(((1.0 - l_1_cond)
              * ((density_IRNormal((torch.tensor(batch)[:,:] - self.thetas[2]) / self.thetas[1]) / self.thetas[1]))), dim=1)

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()

def exp_convergence():
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
            bad=True
        bad = True
        if bad:
            plt.plot(guesses[:, 0], color=c, label="recurser{}".format(i))
            plt.plot(np.ones_like(guesses[:, 0]) * sample_params[0], color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('exp_converge_success.png')
    plt.show()

def exp_batch_convergence():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for i in range(10):
        c = next(color)
        guesses, sample_params = batch_converge_success(i)
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
    plt.savefig('exp_batch_stability.png')
    plt.show()

def exp_batch_performance():
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
        guesses, sample_params = batch_converge_success(i)
        end_time2 = time.time()
        execution_time2 = end_time2 - start_time2
        execution_times2.append(execution_time2)

    mean1 = np.mean(execution_times1)
    mean2 = np.mean(execution_times2)
    x = round(mean1/mean2, 2)
    plt.axhline(mean1, color='gray', linestyle='--')
    plt.axhline(mean2, color='gray', linestyle='--', label="{}x faster".format(x))
    plt.scatter(recurser, execution_times1, marker='o', label="Sequential execution")
    plt.scatter(recurser, execution_times2, marker='+', label="Batch execution")
    plt.xlabel("Recurser")
    plt.ylabel("Execution time (s)")
    plt.legend()
    plt.savefig('exp_batch_performance.png')
    plt.show()


def exp_batch_stability():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for i in range(10):
        c = next(color)
        guesses1, sample_params1 = converge_success(i)
        guesses2, sample_params2 = batch_converge_success(i)
        diff = np.sum(guesses1[:, 0] - guesses2[:, 0])
        plt.plot(guesses1[:, 0], color=c, label="Difference {}".format(diff))
        #plt.plot(guesses2[:, 0], color=c)
        plt.plot(np.ones_like(guesses1[:, 0]) * sample_params1[0], color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Difference")
    plt.legend()
    plt.savefig('exp_batch_stability.png')
    plt.show()

#exp_convergence()
#exp_batch_convergence()
exp_batch_performance()
#exp_batch_stability()
