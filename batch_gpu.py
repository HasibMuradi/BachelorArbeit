import subprocess
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


def converge_success(i, size, epoch):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_sequential()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(size):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / size, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    number_of_batch = []
    for epo in range(epoch):
        start_time = time.time()
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            likelihood.backward(retain_graph=True)
        optimizer.step()
        optimizer.zero_grad()
        end_time = time.time()
        number_of_batch.append(end_time - start_time)

    return number_of_batch

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


def converge_success_minibatch(i, size, epoch, bsize):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_minibatch()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(size):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / size, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    number_of_batch = []
    for epo in range(epoch):
        start_time = time.time()
        for n in range(int(size/bsize)):
            likelihood = -torch.log(gausslist(samples[n*bsize: n*bsize+bsize]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        end_time = time.time()
        number_of_batch.append(end_time - start_time)

    return number_of_batch

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
            print(batch_matrix[i])
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


def converge_success_minibatch_gpu(i, size, epoch, bsize):
    print("hi")
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_minibatch_gpu()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(size):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / size, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    # Define the length of longest sample
    max_sample_length = max([len(sample) for sample in samples])
    # Pad the samples to match the length of the longest sample
    batch_matrix = []
    index_matrix = []
    for i in range(len(samples)):
        batch_matrix.append(samples[i] + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
        index_matrix.append([torch.tensor(1)] * len(samples[i]) + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
    # Convert lists to tensors
    batch_matrix = torch.tensor(batch_matrix)
    index_matrix = torch.tensor(index_matrix)
    number_of_batch = []

    # Move tensors to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_matrix = batch_matrix.to(device)
    index_matrix = index_matrix.to(device)
    gausslist = gausslist.to(device)

    for epo in range(epoch):
        start_time = time.time()
        for n in range(int(size/bsize)):
            likelihood = -torch.log(gausslist(batch_matrix[n * bsize: n * bsize + bsize], index_matrix[n * bsize: n * bsize + bsize]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
        end_time = time.time()
        number_of_batch.append(end_time - start_time)

    return number_of_batch

class Main_minibatch_gpu(Module):
    def forward(self, batch_matrix, index_matrix):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        # Calculate batch_matrix elements
        batch_matrix = (((1.0 - l_1_cond) * (density_IRNormal((batch_matrix - self.thetas[2]) / self.thetas[1]) / self.thetas[1]))
                        * index_matrix) + (1 - index_matrix)

        # Calculate product along dim=1
        return torch.prod(batch_matrix, dim=1) * l_1_cond


    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()

def exp_performance_sequential_minibatch(random_variable, sample_size, epoch, batch_size):

    epoch_list1 = converge_success(random_variable, sample_size, epoch)
    print("Hey")
    epoch_list2 = converge_success_minibatch(random_variable, sample_size, epoch, batch_size)

    mean1 = np.mean(epoch_list1)
    mean2 = np.mean(epoch_list2)

    x = round(mean1 / mean2, 2)
    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x))
    plt.plot(np.linspace(1, 10, 10), epoch_list1, marker='o', color='blue', label="Single-Processing")
    plt.plot(np.linspace(1, 10, 10), epoch_list2, marker='o', color='green', label="Multi-Processing")
    plt.xlabel("Sample Size")
    plt.ylabel("Execution time (s)")
    plt.title("MGD vs. GD training time of model on samples with batch size of.")
    plt.legend()
    plt.show()

def exp_performance_sequential_minibatchInGPU(random_variable, sample_size, epoch, batch_size):

    epoch_list1 = converge_success(random_variable, sample_size, epoch)
    print("Hey")
    epoch_list2 = converge_success_minibatch_gpu(random_variable, sample_size, epoch, batch_size)

    mean1 = np.mean(epoch_list1)
    mean2 = np.mean(epoch_list2)

    x = round(mean1 / mean2, 2)
    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x))
    plt.plot(np.linspace(1, 10, 10), epoch_list1, marker='o', color='blue', label="Single-Processing")
    plt.plot(np.linspace(1, 10, 10), epoch_list2, marker='o', color='green', label="Multi-Processing")
    plt.xlabel("Sample Size")
    plt.ylabel("Execution time (s)")
    plt.title("MGD vs. GD training time of model on samples with batch size of.")
    plt.legend()
    plt.show()

def exp_performance_minibatch_minibatchInGPU(random_variable, sample_size, epoch, batch_size):

    epoch_list1 = converge_success_minibatch(random_variable, sample_size, epoch, batch_size)
    print("Hey")
    epoch_list2 = converge_success_minibatch_gpu(random_variable, sample_size, epoch, batch_size)

    mean1 = np.mean(epoch_list1)
    mean2 = np.mean(epoch_list2)

    x = round(mean1 / mean2, 2)
    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x))
    plt.plot(np.linspace(1, 10, 10), epoch_list1, marker='o', color='blue', label="Single-Processing")
    plt.plot(np.linspace(1, 10, 10), epoch_list2, marker='o', color='green', label="Multi-Processing")
    plt.xlabel("Sample Size")
    plt.ylabel("Execution time (s)")
    plt.title("MGD vs. GD training time of model on samples with batch size of.")
    plt.legend()
    plt.show()

def exp_performance_all(random_variable, sample_size, epoch, batch_size):

    epoch_list1 = converge_success(random_variable, sample_size, epoch)
    epoch_list2 = converge_success_minibatch(random_variable, sample_size, epoch, batch_size)
    print("Hey")
    epoch_list3 = converge_success_minibatch_gpu(random_variable, sample_size, epoch, batch_size)

    mean1 = np.mean(epoch_list1)
    mean2 = np.mean(epoch_list2)
    mean3 = np.mean(epoch_list2)

    x1 = round(mean1 / mean2, 2)
    x2 = round(mean1 / mean3, 2)
    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x1))
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x2))
    plt.plot(np.linspace(1, 10, 10), epoch_list1, marker='o', color='blue', label="Single-Processing")
    plt.plot(np.linspace(1, 10, 10), epoch_list2, marker='o', color='green', label="Multi-Processing")
    plt.plot(np.linspace(1, 10, 10), epoch_list3, marker='o', color='green', label="Multi-Processing in GPU")
    plt.xlabel("Sample Size")
    plt.ylabel("Execution time (s)")
    plt.title("MGD vs. GD training time of model on samples with batch size of.")
    plt.legend()
    plt.show()

def exp_performance_minibatch_10000_samples():

    epoch_list1 = converge_success(0, 10000, 10, 100)
    print("Hey")
    epoch_list2 = converge_success_minibatch_gpu(0, 10000, 10, 100)

    mean1 = np.mean(epoch_list1)
    mean2 = np.mean(epoch_list2)

    x = round(mean1 / mean2, 2)
    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='red', linestyle='dashed', label="{}x faster".format(x))
    plt.plot(np.linspace(1, 10, 10), epoch_list1, marker='o', color='blue', label="Single-Processing")
    plt.plot(np.linspace(1, 10, 10), epoch_list2, marker='o', color='green', label="Multi-Processing")
    plt.xlabel("Sample Size")
    plt.ylabel("Execution time (s)")
    plt.title("MGD vs. GD training time of model on 10000 samples with batch size of 100. MGD training is run on GPU 'NVIDIA GeForce MX250'.")
    plt.legend()
    plt.savefig("matrix_batch_gpu.png")
    plt.show()



#exp_performance_minibatch_10000_samples()
#converge_success_minibatch_gpu(0, 500, 10, 100)
