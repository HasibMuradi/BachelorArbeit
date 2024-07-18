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


def converge_success_columnwise(random_seed, samples_length, epoch, batch_size):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(random_seed)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_columnwise()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(samples_length):
        samples.append(gausslist.generate())

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01/samples_length, momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    final_list = []
    for n in range(int(samples_length/batch_size)):
        max_sample_length = max([len(sample) for sample in samples[batch_size*n: batch_size*n+batch_size]])
        index = []
        list = []
        for i in range(1, max_sample_length + 1):
            tmp_index = []
            tmp_list = []
            for j in range(batch_size):
                if len(samples[batch_size*n+j]) >= i:
                    tmp_index.append(j)
                    tmp_list.append(samples[batch_size*n+j][i-1])
            if tmp_list:
                index.append(tmp_index)
                list.append(tmp_list)
        final_list.append([index, list])

    guesses = []
    execution = []
    for epo in range(epoch):
        for i in range(len(final_list)):
            start_time = time.time()
            likelihood = -torch.log(gausslist(batch_size, final_list[i]))
            end_time = time.time()
            if execution:
                execution.append((end_time - start_time) + execution[epo*len(final_list) + (i - 1)])
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

class Main_columnwise(Module):
    def forward(self, batch, lst):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        result = torch.ones(batch)
        for i in range(len(lst[0])):
            result[lst[0][i]] *= ((1.0 - l_1_cond) * (density_IRNormal((tensor(lst[1][i])
                                    - self.thetas[2]) / self.thetas[1]) / self.thetas[1]))

        return result * l_1_cond


    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()



def exp_performance():

    guesses1, sample_params1, execution1 = Sequential.converge_success(21, 500, 10)
    guesses2, sample_params2, execution2 = converge_success_columnwise(21, 500, 10, 100)

    fig, ax = plt.subplots()

    ax.axhline(sample_params1[0], color="gray", linestyle="dashed")
    ax.plot(execution1, guesses1[:, 0], color="blue", marker="s", label="BGD")
    ax.plot(execution2, guesses2[:, 0], color="orange", marker="^", label="MGD")
    ax.set_title("BGD vs MGD with column-wise approach")
    ax.set_ylabel("Training loss")
    ax.set_xlabel("Training time (s)")
    ax.set_xlim([-0.2, 10])
    ax.set_ylim([0, 1])
    ax.grid(True)
    ax.legend()

    bbox = (0, 0.05, 1, 1)
    axin = inset_axes(ax, width="40%", height="40%", loc='lower right', bbox_to_anchor=bbox, bbox_transform=ax.transAxes)
    axin.set_xlim([0, 0.3])
    axin.set_ylim([0, 1])
    axin.axhline(sample_params2[0], color="gray", linestyle="dashed")
    axin.plot(execution2, guesses2[:, 0], color="orange", marker="^", label='MGD')
    axin.grid(True)
    axin.legend(loc='lower right')

    plt.savefig("columnwise_performance.png")
    plt.show()

def exp_convergence():

    guesses1, sample_params1, execution1 = Sequential.converge_success(21, 500, 10)
    guesses2, sample_params2, execution2 = converge_success_columnwise(21, 500, 10,100)
    plt.plot(np.linspace(0, 10, 10), guesses1[:, 0], color="blue")
    plt.plot(np.linspace(0, 10, 50), guesses2[:, 0], color="violet")
    plt.axhline(sample_params1[0], color="gray", linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel(r"Learning parameter ($\theta_0$)")
    plt.legend()
    plt.savefig("clustering_convergence.png")
    plt.show()


exp_performance()
#exp_convergence()

