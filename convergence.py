import time
from datetime import datetime

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
    return guesses, sample_params

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

    guesses = []
    for epo in range(epoch):
        for iter in range(int(size/bsize)):
            likelihood = -torch.log(gausslist(samples[iter*bsize: iter*bsize+bsize]))
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



"""def converge_success_minibatch_gpu(i, size, epoch, bsize):
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

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_matrix = batch_matrix.to(device)
    index_matrix = index_matrix.to(device)
    gausslist = gausslist.to(device)

    guesses = []
    for epo in range(epoch):
        for iter in range(int(size/bsize)):
            likelihood = -torch.log(gausslist(batch_matrix[iter*bsize: iter*bsize+bsize], index_matrix[iter*bsize: iter*bsize+bsize]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])

        print("Iteration report: ", epo)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params

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
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()"""


def exp_convergence_sequential_minibatch(random_variable, sample_size, epoch, batch_size):

    guesses1, sample_params1 = converge_success(random_variable, sample_size, epoch)
    guesses2, sample_params2 = converge_success_minibatch(random_variable, sample_size, epoch, batch_size)

    if epoch == 1:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], marker="o", color='blue', label="BGD")
    else:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], color='blue', label="BGD")
    plt.plot(np.linspace(0, epoch, int(sample_size/batch_size)*epoch), guesses2[:, 0], color="green", label="MGD")
    plt.axhline(sample_params1[0], color='gray', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence of BGD vs. MGD with {} samples, \n{} batch size and {} epochs.".format(sample_size, batch_size, epoch), fontsize=10)
    plt.legend()
    plt.show()

"""def exp_convergence_sequential_minibatchOnGPU(random_variable, sample_size, epoch, batch_size):

    guesses1, sample_params1 = converge_success(random_variable, sample_size, epoch)
    guesses2, sample_params2 = converge_success_minibatch_gpu(random_variable, sample_size, epoch, batch_size)

    if epoch == 1:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], marker="o", color="blue", label="BGD")
    else:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], color="blue", label="BGD")
    plt.plot(np.linspace(0, epoch, int(sample_size/batch_size)*epoch), guesses2[:, 0], color="orange", label="MGD on GPU")
    plt.axhline(sample_params1[0], color='gray', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence of BGD vs. MGD with {} samples, \n{} batch size and {} epochs.".format(sample_size, batch_size, epoch), fontsize=10)
    plt.legend()
    plt.show()

def exp_convergence_minibatch_minibatchOnGPU(random_variable, sample_size, epoch, batch_size):

    guesses1, sample_params1 = converge_success_minibatch(random_variable, sample_size, epoch, batch_size)
    guesses2, sample_params2 = converge_success_minibatch_gpu(random_variable, sample_size, epoch, batch_size)

    plt.plot(np.linspace(0, epoch, int(sample_size/batch_size)*epoch), guesses1[:, 0], color="green", label="MGD")
    plt.plot(np.linspace(0, epoch, int(sample_size/batch_size)*epoch), guesses2[:, 0], color="orange", label="MGD on GPU", linestyle="dotted")
    plt.axhline(sample_params1[0], color='gray', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence of MGD vs. MGD on GPU with {} samples, \n{} batch size and {} epochs.".format(sample_size, batch_size, epoch), fontsize=10)
    plt.legend()
    plt.show()

def exp_convergence_all(random_variable, sample_size, epoch, batch_size):

    guesses1, sample_params1 = converge_success(random_variable, sample_size, epoch)
    guesses2, sample_params2 = converge_success_minibatch(random_variable, sample_size, epoch, batch_size)
    guesses3, sample_params3 = converge_success_minibatch_gpu(random_variable, sample_size, epoch, batch_size)

    if epoch == 1:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], marker="o", color='blue', label="BGD")
    else:
        plt.plot(np.linspace(1, epoch, epoch), guesses1[:, 0], color='blue', label="BGD")
    plt.plot(np.linspace(0, epoch, int(sample_size/batch_size)*epoch), guesses2[:, 0], color="green", label="MGD")
    plt.plot(np.linspace(0, epoch, int(sample_size/batch_size)*epoch), guesses3[:, 0], color="orange", label="MGD on GPU", linestyle="dotted")
    plt.axhline(sample_params1[0], color='gray', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Convergence of BGD vs. MGD vs. MGD on GPU with {} samples, \n{} batch size and {} epochs.".format(sample_size, batch_size, epoch), fontsize=10)
    plt.legend()
    plt.show()"""


#exp_convergence_matrix_gpu()
#exp_convergence_all(2,500,10,100)
