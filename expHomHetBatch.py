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

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)

    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main1()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    """for n in range(500):
        x = gausslist.generate()
        samples.append(x)
        print(x)"""

    for i in range(10):
        for j in range(50):
            tmp = []
            for k in range(i):
                tmp.append(tensor(rand()*2, requires_grad=True))
            samples.append(tmp)

    for x in samples:
        print(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()
    guesses = []
    for epoch in range(20):
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
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return (guesses, sample_params)

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

def minibatch_converge_success(i):
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
        sample = gausslist.generate()
        flag = False
        index = 0
        for batch in batches_of_samples:
            if (len(sample) < len(batch[0])):
                break
            else:
                if (len(sample) == len(batch[0])):
                    batch.append(sample)
                    flag = True
            index += 1
        if (flag == False):
            batches_of_samples.insert(index, [sample])

    sample_length = 0
    for batch in batches_of_samples:
        sample_length += len(batch)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / sample_length, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    for epoch in range(20):
        for batch in batches_of_samples:
            likelihood = -torch.log(gausslist(batch))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])

        print("iteration report:", epoch)
        print("\t", gausslist.thetas.grad)

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

        indices = []
        flattened_list = []
        for i in range(len(batch)):
            if batch[i]:
                indices += [i] * len(batch[i])
                flattened_list += batch[i]

        return ((torch.ones(len(batch))
                 .index_reduce_(0, tensor(indices, dtype=torch.int32), ((1.0 - l_1_cond) * (density_IRNormal((tensor(flattened_list) - self.thetas[2]) / self.thetas[1]) / self.thetas[1])), 'prod'))
                * l_1_cond)

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()


def exp_convergence():

    guesses, sample_params = converge_success(0)
    guesses_batch, sample_params_batch = minibatch_converge_success(0)

    plt.plot(guesses[:, 0], color="blue", label="Unsorted")
    plt.plot(guesses_batch[:, 0], color="red", label="Sorted")
    plt.axhline(sample_params[0], color="gray", linestyle="dashed")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.savefig('exp_conv.png')
    plt.show()


#exp_convergence()
converge_success(0)



