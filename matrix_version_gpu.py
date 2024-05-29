

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor
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
    return [np.random.rand() * 0.6 + 0.2, np.random.rand() + 0.2, np.random.rand()]



def converge_success_matrix_gpu(i):
    #variant one: initialize seed only here, set to i.
    #variant two: set to 1 here, set to i thereafter.
    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(i)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2]]
    gausslist = Main_matrix_gpu()

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

    # Define the length of longest sample
    max_sample_length = max([len(sample) for sample in samples])
    # Pad the samples to match the length of the longest sample
    batch_matrix = []
    index_matrix = []
    for i in range(len(samples)):
        batch_matrix.append(samples[i] + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
        index_matrix.append([torch.tensor(1)] * len(samples[i]) + [torch.tensor(0)] * max(0, max_sample_length - len(samples[i])))
    print("bm: ", batch_matrix)
    # Convert lists to tensors
    batch_matrix = torch.tensor(batch_matrix)
    index_matrix = torch.tensor(index_matrix)

    # Move tensors to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_matrix = batch_matrix.to(device)
    index_matrix = index_matrix.to(device)
    gausslist = gausslist.to(device)

    for epoch in range(10):
        for n in range(10):
            likelihood = -torch.log(gausslist(batch_matrix[n*50: n*50+50], index_matrix[n*50: n*50+50]))
            likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

        print("iteration report: ", epoch)
        print("\t", gausslist.thetas)
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, sample_params

class Main_matrix_gpu(Module):
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



def exp_convergence_matrix_gpu():
    color = iter(plt.cm.rainbow(np.linspace(0, 1, 10)))
    for i in range(10):
        c = next(color)
        guesses, sample_params = converge_success_matrix_gpu(i)
        tgt = sample_params[0]
        graph = guesses[:, 0]
        bad = False
        graph = graph - np.ones_like(guesses[:, 0]) * tgt
        if graph[0] > 0 and graph[-1] < 0:
            bad = True
        if graph[0] < 0 and graph[-1] > 0:
            bad = True
        if abs(graph[0]) < abs(graph[-1]):
            bad = True
        bad = True
        if bad:
            plt.plot(guesses[:, 0], color=c, label="recurser{}".format(i))
            plt.plot(np.ones_like(guesses[:, 0]) * sample_params[0], color=c, linestyle="dashed")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig('matrix_version_gpu_minibatch_convergence.png')
    plt.show()


exp_convergence_matrix_gpu()
