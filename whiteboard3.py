import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor, random
import numpy as np
import matplotlib.pyplot as plt

import random

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


def converge_success_matrix(i):
    #variant one: initialize seed only here, set to i.
    #variant two: set to 1 here, set to i thereafter.
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_matrix()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))
    print("\tSample Params: ", gausslist.thetas)

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    print("\t", gausslist.thetas)
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    batch = []
    index = []
    for n in range(int(len(samples) / 100)):
        max_sample_length = max([len(sample) for sample in samples[100*n: 100*n+100]])
        batch_matrix = []
        index_matrix = []
        for i in range(100*n, 100*n+100):
            batch_matrix.append(samples[i] + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
            index_matrix.append([torch.tensor(1)] * len(samples[i]) + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
        batch.append(batch_matrix)
        index.append(index_matrix)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(len(index)):
        batch[i] = tensor(batch[i]).to(device)
        index[i] = tensor(index[i]).to(device)
    gausslist = gausslist.to(device)

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for i in range(len(index)):
            likelihood = -torch.log(gausslist(index[i], batch[i]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
        end_time = time.time()
        execution.append(end_time-start_time)
        print("iteration report: ", epoch)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params, execution

class Main_matrix(Module):
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

        return torch.prod((((1.0 - l_1_cond) * (density_IRNormal((batch_matrix - self.thetas[2]) / self.thetas[1]) / self.thetas[1])) * index_matrix) + (1 - index_matrix),
                          dim=1) * l_1_cond

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()



def converge_success_vertical(i):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_vertical()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    final_index = []
    final_list = []
    for n in range(int(len(samples) / 100)):
        max_sample_length = max([len(sample) for sample in samples[100 * n: 100 * n + 100]])
        index = []
        lst = []
        for i in range(1, max_sample_length + 1):
            tmp_index = []
            tmp_list = []
            for j in range(100):
                if len(samples[100 * n + j]) >= i:
                    tmp_index.append(j)
                    tmp_list.append(samples[100 * n + j][i - 1])
            if tmp_list:
                index.append(tmp_index)
                lst.append(tmp_list)
        final_index.append(index)
        final_list.append(lst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(len(final_index)):
        for j in range(len(final_index[i])):
            final_index[i][j] = tensor(final_index[i][j]).to(device)
            final_list[i][j] = tensor(final_list[i][j]).to(device)
    gausslist = gausslist.to(device)

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for i in range(len(final_index)):
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            for j in range(len(final_index[i])):
                final_index[i][j] = tensor(final_index[i][j]).to(device)
                final_list[i][j] = tensor(final_list[i][j]).to(device)
            gausslist = gausslist.to(device)
            likelihood = -torch.log(gausslist(final_index[i], final_list[i]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
        end_time = time.time()
        execution.append(end_time - start_time)
        print("iteration report: ", epoch)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params, execution

class Main_vertical(Module):
    def forward(self, index, lst):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        result = torch.ones(100)
        for i in range(len(index)):
            result.index_put_((index[i],),
                              ((1.0 - l_1_cond) * (density_IRNormal((lst[i] - self.thetas[2]) / self.thetas[1]) / self.thetas[1]))
                              * result.index_select(0, index[i]))

        return result * l_1_cond


    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()



"""def converge_success_horizontal(i):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_horizontal()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    final_list = []
    for n in range(int(len(samples) / 100)):
        max_sample_length = max([len(sample) for sample in samples[100 * n: 100 * n + 100]])
        lst = []
        for i in range(1, max_sample_length + 1):
            tmp_index = []
            tmp_list = []
            for j in range(100):
                if len(samples[100 * n + j]) == i:
                    tmp_index.append(j)
                    tmp_list.append(samples[100 * n + j])
            if tmp_list:
                lst.append([tmp_index, tmp_list])
        final_list.append(lst)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for elem in final_list:
        elem[0] = tensor(elem[0]).to(device)
        elem[1] = tensor(elem[1]).to(device)
    gausslist = gausslist.to(device)

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for elem in final_list:
            likelihood = -torch.log(gausslist(elem))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
        end_time = time.time()
        execution.append(end_time - start_time)
        print("iteration report: ", epoch)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params, execution


class Main_horizontal(Module):
    def forward(self, final_list):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)

        result = torch.ones(100)
        for elem in final_list:
            result.index_put_((torch.tensor(elem[0]).to(torch.int64),),
                              torch.prod(((1.0 - l_1_cond) * (density_IRNormal((torch.tensor(elem[1]) - self.thetas[2]) / self.thetas[1]) / self.thetas[1])), dim=1))

        return result * l_1_cond

    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()"""


def converge_success_flattening(i):
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_flattening()

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for n in range(500):
        x = gausslist.generate()
        samples.append(x)

    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    indices_lists = []
    flattened_lists = []
    for i in range(int(len(samples) / 100)):
        indices = []
        flattened_list = []
        j = 0
        for sample in samples[100 * i: 100 * i + 100]:
            if sample:
                indices += [j] * len(sample)
                flattened_list += sample
            j += 1
        if indices:
            indices_lists.append(indices)
            flattened_lists.append(flattened_list)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(len(indices_lists)):
        indices_lists[i] = tensor(indices_lists[i]).to(device)
        flattened_lists[i] = tensor(flattened_lists[i]).to(device)
    gausslist = gausslist.to(device)

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for i in range(len(indices_lists)):
            likelihood = -torch.log(gausslist(indices_lists[i], flattened_lists[i]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
        end_time = time.time()
        execution.append(end_time - start_time)
        print("iteration report: ", epoch)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params, execution


class Main_flattening(Module):
    def forward(self, indices, flattened_list):
        if (1.0 >= self.thetas[0]):
            l_4_high = self.thetas[0]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)


        return ((torch.ones(100)
                 .index_reduce_(0, indices, ((1.0 - l_1_cond) * (density_IRNormal((flattened_list - self.thetas[2]) / self.thetas[1]) / self.thetas[1])), 'prod'))
                * l_1_cond)


    def generate(self):
        if (rand() >= self.thetas[0]):
            return []
        else:
            return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()


def exp_performance():

    recurser = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    guesses1, sample_params1, execution1 = converge_success_matrix(0)
    guesses2, sample_params2, execution2 = converge_success_vertical(0)
    #guesses3, sample_params3, execution3 = converge_success_horizontal(0)
    guesses4, sample_params4, execution4 = converge_success_flattening(0)

    mean1 = np.mean(execution1)
    mean2 = np.mean(execution2)
    #mean3 = np.mean(execution3)
    mean4 = np.mean(execution4)

    plt.axhline(mean1, color='red', linestyle='dashed')
    plt.axhline(mean2, color='blue', linestyle='dashed')
    #plt.axhline(mean3, color='green', linestyle='dashed')
    plt.axhline(mean4, color='orange', linestyle='dashed')
    plt.scatter(recurser, execution1, marker='o', color='red', label="Matrix")
    plt.scatter(recurser, execution2, marker='o', color='blue', label="Column by column execution")
    #plt.scatter(recurser, execution3, marker='o', color='green', label="Row by row execution")
    plt.scatter(recurser, execution4, marker='o', color='orange', label="Flattening")
    plt.xlabel("Epoch")
    plt.ylabel("Execution time (s)")
    plt.legend()
    plt.title("Without preprocessing on GPU")
    plt.savefig('without_preprocessing_GPU.png')
    plt.show()

def exp_convergence():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))

    for i in range(10):
        c = next(color)
        guesses, sample_params, execution = converge_success_flattening(i)
        plt.plot(np.linspace(0, 10, 50), guesses[:, 0], color=c, label="Recurser: {}".format(i))
        plt.axhline(sample_params[0], color='gray', linestyle='dashed')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


#exp_performance()
exp_convergence()
#converge_success_matrix(0)

