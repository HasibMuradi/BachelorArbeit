import itertools
import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor, random
import numpy as np
import matplotlib.pyplot as plt

import Sequential
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from matplotlib.patches import Rectangle

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


def converge_success_padding(random_seed, samples_length, epoch, batch_size):

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(random_seed)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_padding()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for _ in itertools.count():
        sample = gausslist.generate()
        if len(sample) == 20:
            samples.append(gausslist.generate())
        if len(samples) == samples_length:
            break

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01/samples_length, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    # Preprocessing for padding
    batch = []
    index = []
    for n in range(int(samples_length/batch_size)):
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
    return guesses, sample_params, execution

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




def converge_success_flattening(random_seed, samples_length, epoch, batch_size):

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(random_seed)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_flattening()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for _ in itertools.count():
        sample = gausslist.generate()
        if len(sample) == 20:
            samples.append(gausslist.generate())
        if len(samples) == samples_length:
            break

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / samples_length, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    indices_lists = []
    flattened_lists = []
    for i in range(int(samples_length/batch_size)):
        indices = []
        flattened_list = []
        j = 0
        for sample in samples[batch_size*i: batch_size*i+batch_size]:
            if sample:
                indices += [j] * len(sample)
                flattened_list += sample
            j += 1
        if indices:
            indices_lists.append(indices)
            flattened_lists.append(flattened_list)

    guesses = []
    execution = []
    for epo in range(epoch):
        for i in range(int(samples_length/batch_size)):
            start_time = time.time()
            likelihood = -torch.log(gausslist(batch_size, tensor(indices_lists[i]), tensor(flattened_lists[i])))
            end_time = time.time()
            if execution:
                execution.append((end_time - start_time) + execution[epo * int(samples_length/batch_size) + (i - 1)])
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
    return guesses, sample_params, execution


class Main_flattening(Module):
    def forward(self, batch, indices, flattened_list):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        return ((torch.ones(batch).index_reduce_(0, indices,
                ((1.0 - l_1_cond) * (density_IRNormal((flattened_list - self.thetas[2])
                    /self.thetas[1]) / self.thetas[1])), 'prod')) * l_1_cond)

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()




def exp_performance():

    guesses1, sample_params1, execution1 = converge_success_padding(21, 500, 40, 100)
    guesses2, sample_params2, execution2 = converge_success_flattening(21, 500, 40, 100)

    plt.plot(execution1, guesses1[:, 0], color="blue", marker="o", label="Padding")
    plt.plot(execution2, guesses2[:, 0], color="violet", marker="o", label="Flattening")
    plt.axhline(sample_params1[0], color="gray", linestyle="dashed")
    plt.xlabel("Execution time")
    plt.ylabel("Theta[0]")
    plt.title("Ablation study of padding vs. flattening approach")
    plt.legend()
    plt.savefig("padFlatAblationStudy.png")
    plt.show()




exp_performance()

