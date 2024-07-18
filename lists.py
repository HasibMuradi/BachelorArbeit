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
        print("Param: ", gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item())
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

        result = torch.ones(len(batch))
        for i in range(len(batch[0])):
            result *= ((1.0 - l_1_cond) * ((density_IRNormal((torch.tensor(batch)[:,i] - self.thetas[2]) / self.thetas[1]) / self.thetas[1])))
        return result * l_1_cond

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()

def exp_conv():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for i in range(10):
        c = next(color)
        guesses, sample_params = converge_success(i)
        plt.plot(guesses[:, 0], color=c, label="recurser{}".format(i))
        plt.plot(sample_params[0], color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.savefig('exp_conv.png')
    plt.show()


#exp_batch()
#exp_minibatch()
exp_conv()
#exp_batch_conv()
#exp_minibatch_conv()


#minibatch_converge_success(1)


