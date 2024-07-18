import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor, random
import numpy as np
import matplotlib.pyplot as plt


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
    return [rand() * 0.6 + 0.2, rand() * 0.4 + 0.2, rand() * 0.2 + 0.2, rand() + 0.2, rand()]


def converge_success(samples, search_params, epoch):

    gausslist = Program1()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.1 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    guesses = []
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                    gausslist.thetas[3].item(), gausslist.thetas[4].item()])
    execution = []
    execution.append(0)
    llh = []
    for epo in range(epoch):
        start_time = time.time()
        for sample in samples:
            likelihood = -torch.log(gausslist(sample))
            llh.append(likelihood.item())
            likelihood.backward(retain_graph=True)
        end_time = time.time()
        execution.append((end_time - start_time) + execution[epo])

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas.grad)
        optimizer.step()
        optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                        gausslist.thetas[3].item(), gausslist.thetas[4].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, execution, llh

class Program1(Module):
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

        return ((l_1_cond * (0.0 if (sample == [])
                else ((density_IRNormal((((sample)[0] - self.thetas[2]) / self.thetas[1])) / self.thetas[1])
                      * (1.0 if ((sample)[1:] == []) else 0.0))))
                + ((1.0 - l_1_cond) * (0.0 if (sample == [])
                    else ((density_IRNormal((((sample)[0] - self.thetas[4]) / self.thetas[3])) / self.thetas[3])
                        * (1.0 if ((sample)[1:] == []) else 0.0)))))

    def generate(self):
        if (rand() >= self.thetas[0]):
            return [((randn() * self.thetas[1]) + self.thetas[2])] + []
        else:
            return [((randn() * self.thetas[3]) + self.thetas[4])] + []



def converge_success_flattening(samples, search_params, epoch, batch_size):

    gausslist = Main_flattening()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.1 / len(samples), momentum=0.001)
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
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                    gausslist.thetas[3].item(), gausslist.thetas[4].item()])
    execution = []
    execution.append(0)
    llh = []
    for epo in range(epoch):
        for i in range(int(len(samples)/batch_size)):
            start_time = time.time()
            likelihood = -torch.log(gausslist(batch_size, tensor(indices_lists[i]), tensor(flattened_lists[i])))
            end_time = time.time()
            execution.append((end_time - start_time) + execution[epo * int(len(samples)/batch_size) + i])
            llh += likelihood.tolist()
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                           gausslist.thetas[3].item(), gausslist.thetas[4].item()])

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, execution, llh

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

        return torch.ones(batch).index_reduce_(0, indices,
                ((l_1_cond * (density_IRNormal(((flattened_list - self.thetas[2]) / self.thetas[1])) / self.thetas[1]))
                + ((1.0 - l_1_cond) * (((density_IRNormal(((flattened_list - self.thetas[4]) / self.thetas[3])) / self.thetas[3]))))),
                'prod')

    def generate(self):
        if (rand() >= self.thetas[0]):
            return [((randn() * self.thetas[1]) + self.thetas[2])] + []
        else:
            return [((randn() * self.thetas[3]) + self.thetas[4])] + []


def exp_convergence():

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(21)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2], sample_params[3], sample_params[4]]
    gausslist = Program1()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for i in range(500):
        samples.append(gausslist.generate())
        #print("Sample: ", samples[i])

    guesses1, execution1, llh1 = converge_success(samples, search_params, 50)
    guesses2, execution2, llh2 = converge_success_flattening(samples, search_params, 50, 100)
    print("n1: ", execution1[len(execution1)-1])
    print("n2: ", execution2[len(execution2)-1])
    print("t: ", execution1[len(execution1)-1]/execution2[len(execution2)-1])
    plt.plot(execution1, guesses1[:, 0], color="blue", label="First program")
    plt.plot(execution1, guesses1[:, 0], color="blue", marker="o")
    plt.plot(execution2, guesses2[:, 0], color="orange", label="Flattening-parallelized first program")
    plt.plot(np.array(execution2)[1::5], guesses2[:, 0][1::5], color="orange", marker="o")
    plt.axhline(sample_params[0], color="gray", linestyle="dashed")
    plt.xlim([0, 40])
    plt.ylim([0, 1])
    plt.title("Performance of first program")
    plt.xlabel("Training time (s)")
    plt.ylabel(r"Learning parameter ($\theta_0$)")
    plt.legend()
    plt.savefig("program1_convergence.png")
    plt.show()


def exp_llh_comparison():

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(21)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2], sample_params[3], sample_params[4]]
    gausslist = Program1()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for i in range(100):
        samples.append(gausslist.generate())
        # print("Sample: ", samples[i])

    guesses1, execution1, llh1 = converge_success(samples, search_params, 1)
    guesses2, execution2, llh2 = converge_success_flattening(samples, search_params, 1, 50)

    x = np.linspace(0, 10, 10)
    y = x
    plt.plot(x, y, zorder=1, color="gray", linestyle="dashed", label='Diagonal line')
    plt.scatter(np.array(llh1), np.array(llh2), zorder=2, color="blue")
    plt.title("Likelihood comparison of first program")
    plt.xlabel("Likelihood of sequential BGD")
    plt.ylabel("Likelihood of flattening-parallelized MGD")
    plt.xlim([0, 8])
    plt.ylim([0, 8])
    plt.grid(True)
    plt.legend()
    plt.savefig("program1_llh_comparison.png")
    plt.show()

exp_convergence()
#exp_llh_comparison()


