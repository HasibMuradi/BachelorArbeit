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

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for n in range(int(len(samples)/100)):
            likelihood = -torch.log(gausslist(samples[n*100: n*100+100]))
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

class Main_matrix(Module):
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

        # Matrix calculation
        max_sample_length = max([len(sample) for sample in batch])
        batch_matrix = []
        index_matrix = []
        for i in range(len(batch)):
            batch_matrix.append(batch[i] + [torch.tensor(0)] * max(0, max_sample_length - len(batch[i])))
            index_matrix.append([torch.tensor(1)] * len(batch[i]) + [torch.tensor(0)] * max(0, max_sample_length - len(batch[i])))
            print(batch_matrix[i])
        batch_matrix = torch.tensor(batch_matrix)
        index_matrix = torch.tensor(index_matrix)

        return torch.prod((((1.0 - l_1_cond) * (density_IRNormal((batch_matrix - self.thetas[2]) / self.thetas[1]) / self.thetas[1])) * index_matrix) + (1-index_matrix),
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

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for n in range(int(len(samples)/100)):
            likelihood = -torch.log(gausslist(samples[n*100: n*100+100]))
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

        # Vertical calculation
        max_sample_length = max([len(sample) for sample in batch])
        index = []
        list = []
        for i in range(1, max_sample_length+1):
            tmp_index = []
            tmp_list = []
            for j in range(len(batch)):
                if len(batch[j]) >= i:
                    tmp_index.append(j)
                    tmp_list.append(batch[j][i-1])
            if tmp_list:
                index.append(tmp_index)
                list.append(tmp_list)

        result = torch.ones(len(batch))
        for i in range(len(index)):
            result[index[i]] *= ((1.0 - l_1_cond) * (density_IRNormal((torch.tensor(list[i]) - self.thetas[2]) / self.thetas[1]) / self.thetas[1]))

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

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for n in range(int(len(samples) / 100)):
            likelihood = -torch.log(gausslist(samples[n * 100: n * 100 + 100]))
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

        # Horizontal calculation
        max_sample_length = max([len(sample) for sample in batch])
        final_list = []
        for i in range(1, max_sample_length + 1):
            tmp_index = []
            tmp_list = []
            for j in range(len(batch)):
                if len(batch[j]) == i:
                    tmp_index.append(j)
                    tmp_list.append(batch[j])
            if tmp_list:
                final_list.append([tmp_index, tmp_list])

        result = torch.ones(len(batch))
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

    guesses = []
    execution = []
    for epoch in range(10):
        start_time = time.time()
        for i in range(int(len(samples)/100)):
            likelihood = -torch.log(gausslist(samples[100*i: 100*i+100]))
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

        """return ((torch.ones(len(batch))
                 .index_reduce_(0, tensor(indices), ((1.0 - l_1_cond) * (density_IRNormal((tensor(flattened_list) - self.thetas[2]) / self.thetas[1]) / self.thetas[1])),'prod'))
                * l_1_cond)"""

        result = torch.ones(len(batch))
        result[tensor(indices)] *= ((1.0 - l_1_cond) * (density_IRNormal((tensor(flattened_list) - self.thetas[2]) / self.thetas[1]) / self.thetas[1]))
        return result * l_1_cond


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
    plt.title("With preprocessing")
    plt.savefig('with_preprocessing.png')
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

