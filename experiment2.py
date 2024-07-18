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



def converge_success_columnwise(random_seed, samples_length, epoch, batch_size, param):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = param
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



def converge_success_padding(random_seed, samples_length, epoch, batch_size, param):

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = param
    np.random.seed(random_seed)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_padding()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(samples_length):
        samples.append(gausslist.generate())

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


def converge_success_flattening(random_seed, samples_length, epoch, batch_size, param):

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = param
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



def exp_convergence():

    guesses1, sample_params1, execution1 = converge_success_columnwise(21, 4000, 1, 100, 0.8)
    guesses2, sample_params2, execution2 = converge_success_padding(21, 4000, 1, 100, 0.8)
    guesses11, sample_params11, execution11 = converge_success_flattening(21, 4000, 1, 100, 0.8)

    guesses3, sample_params3, execution3 = converge_success_columnwise(21, 4000, 1, 400, 0.8)
    guesses4, sample_params4, execution4 = converge_success_padding(21, 4000, 1, 400, 0.8)
    guesses12, sample_params12, execution12 = converge_success_flattening(21, 4000, 1, 400, 0.8)

    guesses5, sample_params5, execution5 = converge_success_columnwise(21, 4000, 1, 100, 0.4)
    guesses6, sample_params6, execution6 = converge_success_padding(21, 4000, 1, 100, 0.4)
    guesses13, sample_params13, execution13 = converge_success_flattening(21, 4000, 1, 100, 0.4)

    guesses7, sample_params7, execution7 = converge_success_columnwise(21, 4000, 1, 100, 0.9)
    guesses8, sample_params8, execution8 = converge_success_padding(21, 4000, 1, 100, 0.9)
    guesses14, sample_params14, execution14 = converge_success_flattening(21, 4000, 1, 100, 0.9)

    plt.plot(execution1, guesses1[:, 0], color="blue", linestyle="-", label="Column-wise: 4000; 100")
    plt.plot(execution2, guesses2[:, 0], color="blue", linestyle="--", label="Padding: 4000; 100")
    plt.plot(execution11, guesses11[:, 0], color="blue", linestyle="-.", label="Flattening: 4000; 100")

    plt.plot(execution3, guesses3[:, 0], color="green", linestyle="-", label="Column-wise: 4000; 400")
    plt.plot(execution4, guesses4[:, 0], color="green", linestyle="--", label="Padding: 4000; 400")
    plt.plot(execution12, guesses12[:, 0], color="green", linestyle="-.", label="Flattening: 4000; 400")

    plt.plot(execution5, guesses5[:, 0], color="orange", linestyle="-", label="Column-wise: 4000; 100; 0.4")
    plt.plot(execution6, guesses6[:, 0], color="orange", linestyle="--", label="Padding: 4000; 100; 0.4")
    plt.plot(execution13, guesses13[:, 0], color="orange", linestyle="-.", label="Flattening: 4000; 100; 0.4")

    plt.plot(execution7, guesses7[:, 0], color="red", linestyle="-", label="Column-wise: 4000; 100; 0.9")
    plt.plot(execution8, guesses8[:, 0], color="red", linestyle="--", label="Padding: 4000; 100; 0.9")
    plt.plot(execution14, guesses14[:, 0], color="red", linestyle="-.", label="Flattening: 4000; 100; 0.9")

    plt.axhline(sample_params1[0], color='gray', linestyle='dashed')
    plt.xlabel("Training time")
    plt.ylabel("Training loss")
    plt.title("Column-wise vs. Padding vs. Flattening")
    plt.legend()
    plt.savefig("experiment3.png")
    plt.show()

exp_convergence()