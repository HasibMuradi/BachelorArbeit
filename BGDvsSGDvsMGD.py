import random

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


def converge_BGD(i, size, epoch):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_BGD()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(size):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / size, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    for epo in range(epoch):
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            likelihood.backward(retain_graph=True)
        print("Epoch report: ", epo)
        print("\t", gausslist.thetas)
        optimizer.step()
        optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params

class Main_BGD(Module):
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


def converge_SGD(i, size, epoch):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_SGD()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(size):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.05 / size, momentum=0.001)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.75)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    for epo in range(epoch):
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
        print("Epoch report: ", epo)
        print("\t", gausslist.thetas)
        scheduler.step()
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params

class Main_SGD(Module):
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


def converge_MGD(i, size, epoch):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_MGD()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(size):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.02 / size, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    for epo in range(epoch):
        for i in range(int(len(samples)/100)):
            likelihood = -torch.log(gausslist(samples[i*100: i*100+100]))
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
    return guesses, sample_params

class Main_MGD(Module):
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
                 .index_reduce_(0, tensor(indices), ((1.0 - l_1_cond) * (density_IRNormal((tensor(flattened_list) - self.thetas[2]) / self.thetas[1]) / self.thetas[1])),'prod'))
                * l_1_cond)

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()


def exp_convergence_BGDvsSGD(random_variable, sample_size, epoch):

    guesses1, sample_params1 = converge_BGD(random_variable, sample_size, epoch)
    guesses2, sample_params2 = converge_SGD(random_variable, sample_size, epoch)
    guesses3, sample_params3 = converge_MGD(random_variable, sample_size, epoch)

    if epoch == 1:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], marker="o", color='blue', label="BGD")
    else:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], color='blue', label="BGD")
    plt.plot(np.linspace(1/sample_size, epoch, sample_size * epoch), guesses2[:, 0], color="green", label="SGD")
    plt.plot(np.linspace(1/int(sample_size/100), epoch, int(sample_size/100) * epoch), guesses3[:, 0], color="red", label="MGD")
    plt.axhline(sample_params1[0], color='gray', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence of BGD vs. SGD vs. MGD")
    plt.legend()
    plt.savefig("BGSvsSGDvsMGD.png")
    plt.show()


exp_convergence_BGDvsSGD(1, 500, 20)
