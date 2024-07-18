import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor, random
import numpy as np
import matplotlib.pyplot as plt

import Sequential
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle
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
    execution = []
    for epo in range(epoch):
        for i in range(len(index)):
            start_time = time.time()
            likelihood = -torch.log(gausslist(tensor(index[i]), tensor(batch[i])))
            end_time = time.time()
            if execution:
                execution.append((end_time - start_time) + execution[epo*len(index) + (i-1)])
            else:
                execution.append(end_time - start_time)
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
    np.random.seed(21)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    while len(samples) != 500:
        sample = gausslist.generate()
        if len(sample) == 10:
            samples.append(sample)

    for j in range(len(samples)):
       print("Sample: ", samples[j])
    print("len: ", len(samples))
    samples_length = [0, 100, 200, 300, 400, 500]
    s1 = []
    for i in range(len(samples)):
        s1.append(samples[i])
    guesses1, execution1 = converge_success_padding(s1, search_params, 1, 100)
    s2 = []
    for i in range(len(samples)):
        s2.append(samples[i] + ([tensor(0)] * 10))
    guesses2, execution2 = converge_success_padding(s2, search_params, 1, 100)
    s3 = []
    for i in range(len(samples)):
        s3.append(samples[i] + ([tensor(0)] * 30))
        print("s3: ", s3[i])
    guesses3, execution3 = converge_success_padding(s3, search_params, 1, 100)
    s4 = []
    for i in range(len(samples)):
        s4.append(samples[i] + ([tensor(0)] * 50))
    guesses4, execution4 = converge_success_padding(s4, search_params, 1, 100)

    mean1 = execution1[len(execution1) - 1]
    mean2 = execution2[len(execution2) - 1]
    mean3 = execution3[len(execution3) - 1]
    mean4 = execution4[len(execution4) - 1]
    print("mean1: ", mean1)
    print("mean2: ", mean2)
    print("mean3: ", mean3)
    print("mean4: ", mean4)
    #print(mean2 / mean1, "x")
    #print(mean3 / mean2, "x")

    plt.plot([0]+execution1, samples_length, color="blue", marker="o", label="w/o padding")
    plt.plot([0]+execution2, samples_length, color="purple", marker="o", label="1x padding")
    plt.plot([0]+execution3, samples_length, color="brown", marker="o", label="3x padding")
    plt.plot([0]+execution4, samples_length, color="orange", marker="o", label="5x padding")
    plt.title("Computational effect of padding approach")
    plt.ylabel("Sample size")
    plt.xlabel("Training time (s)")
    plt.xlim([0, 0.1])
    plt.ylim([0, 600])
    plt.grid(True)
    plt.legend()
    plt.savefig("padding_experiment.png")
    plt.show()

class Main(Module):
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


exp_performance()
#exp_convergence()

