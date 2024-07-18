import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor, random
import numpy as np
import matplotlib.pyplot as plt

import Sequential
import Padding
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


def converge_success(random_seed, samples_length, epoch):

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(random_seed)

    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(samples_length):
        samples.append(gausslist.generate())

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / samples_length, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    execution = []
    execution.append(0)
    for epo in range(epoch):
        start_time = time.time()
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            likelihood.backward(retain_graph=True)
        end_time = time.time()
        execution.append((end_time - start_time) + execution[epo])

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas.grad)
        optimizer.step()
        optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params, execution


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
    for n in range(samples_length):
        samples.append(gausslist.generate())

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
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    execution = []
    execution.append(0)
    for epo in range(epoch):
        for i in range(int(samples_length/batch_size)):
            start_time = time.time()
            likelihood = -torch.log(gausslist(batch_size, tensor(indices_lists[i]), tensor(flattened_lists[i])))
            end_time = time.time()
            execution.append((end_time - start_time) + execution[epo * int(samples_length/batch_size) + i])
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

    guesses1, sample_params1, execution1 = converge_success(21, 1200, 10)
    guesses2, sample_params2, execution2 = converge_success_flattening(21, 1200, 10, 400)
    x = 0
    if (execution2[len(execution2) - 1] > 0.08 and execution2[len(execution2) - 1] < 0.1):
        x = 0.1
    elif (execution2[len(execution2) - 1] > 0.06 and execution2[len(execution2) - 1] < 0.08):
        x = 0.08
    elif (execution2[len(execution2) - 1] > 0.04 and execution2[len(execution2) - 1] < 0.06):
        x = 0.06
    else:
        x = 0.04
    xx = 0
    if (execution1[len(execution1) - 1] > 30 and execution1[len(execution1) - 1] < 40):
        xx = 40
    elif (execution1[len(execution1) - 1] > 20 and execution1[len(execution1) - 1] < 30):
        xx = 30
    elif (execution1[len(execution1) - 1] > 40 and execution1[len(execution1) - 1] < 50):
        xx = 50
    else:
        xx = 20
    mean1 = execution1[len(execution1) - 1]
    mean2 = execution2[len(execution2) - 1]
    print("mean1: ", mean1)
    print("mean2: ", mean2)
    print(mean1/mean2, "x")

    fig, ax = plt.subplots()

    ax.axhline(sample_params1[0], color="gray", linestyle="dashed")
    ax.plot(execution1, guesses1[:, 0], color="blue", label="Sequential BGD")
    ax.scatter(np.array(execution1)[1::1], guesses1[:, 0][1::1], color="blue", marker="o")
    ax.plot(execution2, guesses2[:, 0], color="orange", label="Flattening-parallelized MGD")
    ax.scatter(np.array(execution2)[3::3], guesses2[:, 0][3::3], color="orange", marker="o")
    ax.set_title("Sequential BGD vs. flattening-parallelized MGD")
    ax.set_ylabel(r"Learning parameter ($\theta_0$)")
    ax.set_xlabel("Training time (s)")
    ax.set_xlim([0, xx])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()

    bbox = (-0.3, 0.05, 1.3, 1)
    axin = inset_axes(ax, width="40%", height="40%", loc='lower right', bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axin.axhline(sample_params2[0], color="gray", linestyle="dashed")
    axin.plot(execution2, guesses2[:, 0], color="orange", label='Flattening-parallelized MGD')
    axin.scatter(np.array(execution2)[3::3], guesses2[:, 0][3::3], color="orange", marker="o")
    axin.set_xlim([0, x])
    axin.set_ylim([0, 1])
    axin.grid(True)
    axin.legend(loc='lower right')

    plt.savefig("flattening_performance.png")
    plt.show()


def exp_convergence():

    guesses1, sample_params1, execution1 = Sequential.converge_success(0, 500, 10)
    guesses2, sample_params2, execution2 = converge_success_flattening(0, 500, 10, 100)
    plt.plot(np.linspace(0, 10, 10), guesses1[:, 0], color="blue")
    plt.plot(np.linspace(0, 10, 50), guesses2[:, 0], color="violet")
    plt.axhline(sample_params1[0], color="gray", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    #plt.savefig("flattening_convergence.png")
    plt.show()

def performance():
    guesses1, sample_params1, execution1 = Sequential.converge_success(21, 500, 10)
    guesses2, sample_params2, execution2 = converge_success_flattening(21, 500, 10, 100)
    plt.plot(execution1, guesses1[:, 0], color="blue")
    plt.plot(execution2, guesses2[:, 0], color="orange")
    plt.axhline(sample_params1[0], color="gray", linestyle="dashed")

    plt.xlabel("Training time (s)")
    plt.ylabel("Parameter theta[0]")
    plt.legend()
    plt.savefig("flattening_perf.png")
    plt.show()

exp_performance()
#exp_convergence()
#performance()
