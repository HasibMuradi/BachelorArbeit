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


def converge_success(random_seed, epoch, training_data_length, test_data_length):

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(random_seed)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]

    gausslist = Main()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    training = []
    test = []
    for _ in range(training_data_length):
        training.append(gausslist.generate())
    for _ in range(test_data_length):
        test.append(gausslist.generate())

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(training), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    for epo in range(epoch):
        for sample in training:
            likelihood = -torch.log(gausslist(sample))
            likelihood.backward(retain_graph=True)

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas.grad)
        optimizer.step()
        optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)

    return gausslist, training, test, search_params

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


def converge_success_sequential(epoch, samples, params):

    gausslist = Main_sequential()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    for epo in range(epoch):
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            likelihood.backward(retain_graph=True)

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas.grad)
        optimizer.step()
        optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return gausslist

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


def converge_success_padding(epoch, batch_size, samples, params):

    gausslist = Main_padding()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(params))
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
    for epo in range(epoch):
        for i in range(len(index)):
            likelihood = -torch.log(gausslist(tensor(index[i]), tensor(batch[i])))
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
    return gausslist

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



def converge_success_flattening(epoch, batch_size, samples, params):

    gausslist = Main_flattening()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    indices_lists = []
    flattened_lists = []
    for i in range(int(len(samples)/batch_size)):
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
    for epo in range(epoch):
        for i in range(int(len(samples)/batch_size)):
            likelihood = -torch.log(gausslist(batch_size, tensor(indices_lists[i]), tensor(flattened_lists[i])))
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
    return gausslist


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


def exp_convergence():

    gausslist, training_data, test_data, params = converge_success(21, 10, 500, 500)

    gausslist_sequential = converge_success_sequential(10, training_data, params)

    gausslist_padding = converge_success_padding(10, 100, training_data, params)
    batch = []
    index = []
    for n in range(int(len(test_data) / 100)):
        max_sample_length = max([len(sample) for sample in test_data[100 * n: 100 * n + 100]])
        batch_matrix = []
        index_matrix = []
        for i in range(100 * n, 100 * n + 100):
            batch_matrix.append(test_data[i] + [torch.tensor(0)] * max(0, max_sample_length - len(test_data[i])))
            index_matrix.append(
                [torch.tensor(1)] * len(test_data[i]) + [torch.tensor(0)] * max(0, max_sample_length - len(test_data[i])))
        batch.append(batch_matrix)
        index.append(index_matrix)

    gausslist_flattening = converge_success_flattening(10, 100, training_data, params)
    indices_lists = []
    flattened_lists = []
    for i in range(int(len(test_data) / 100)):
        indices = []
        flattened_list = []
        j = 0
        for sample in test_data[100 * i: 100 * i + 100]:
            if sample:
                indices += [j] * len(sample)
                flattened_list += sample
            j += 1
        if indices:
            indices_lists.append(indices)
            flattened_lists.append(flattened_list)

    gausslist_likelihoods = 0
    gausslist_sequential_likelihoods = 0
    gausslist_padding_likelihoods = 0
    gausslist_flattening_likelihoods = 0
    for sample in test_data:
        gausslist_likelihoods += -torch.log(gausslist(sample))
        gausslist_sequential_likelihoods += -torch.log(gausslist_sequential(sample))
    for i in range(int(len(test_data) / 100)):
        gausslist_padding_likelihoods += -sum(torch.log(gausslist_padding(tensor(index[i]), tensor(batch[i]))))
        gausslist_flattening_likelihoods += -sum(torch.log(gausslist_flattening(100, tensor(indices_lists[i]), tensor(flattened_lists[i]))))

    print("Likelihoods of BGD as base: ", gausslist_likelihoods)
    print("Likelihoods of BGD as comparator: ", gausslist_sequential_likelihoods)
    print("Likelihoods of MGD using padding approach : ", gausslist_padding_likelihoods)
    print("Likelihoods of MGD using flattening approach : ", gausslist_flattening_likelihoods)


exp_convergence()