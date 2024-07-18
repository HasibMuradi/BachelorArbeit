import time

import torch.nn.parameter
from torch.nn import Module
from torch import optim, tensor, random
import numpy as np
import matplotlib.pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)

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
    """return [rand() * 0.2 + 0.1, rand() * 0.6,
            rand() * 0.4 + 1.0, rand() * 0.4,
            rand() * 0.2 + 0.2, rand() * 0.2,
            rand() - 1.0, rand() * 0.2 + 0.6]"""
    return [rand(), rand(), rand(), rand(),
            rand(), rand(), rand(), rand()]


def converge_success(samples, search_params, epoch):

    gausslist = Program4()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(search_params), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    for param in gausslist.parameters():
        param.data = param.data.double()
        if param.grad is not None:
            param.grad.data = param.grad.data.double()
    gausslist = gausslist.double()
    guesses = []
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                    gausslist.thetas[3].item(), gausslist.thetas[4].item(), gausslist.thetas[5].item(),
                    gausslist.thetas[6].item(), gausslist.thetas[7].item()])
    execution = []
    execution.append(0)
    llh = []
    for epo in range(epoch):
        start_time = time.time()
        for sample in samples:
            likelihood = gausslist(sample)
            #print("lk1: ", likelihood)
            #likelihood = -torch.log(gausslist(sample))
            print("l: ", likelihood)
            llh.append(likelihood.item())
            #likelihood.backward(retain_graph=True)
        end_time = time.time()
        execution.append((end_time - start_time) + execution[epo])

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas.grad)
        #optimizer.step()
        #optimizer.zero_grad()
        guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                        gausslist.thetas[3].item(), gausslist.thetas[4].item(), gausslist.thetas[5].item(),
                        gausslist.thetas[6].item(), gausslist.thetas[7].item()])
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, execution, llh

class Program4(Module):
    def forward(self, sample):
        if (1.0 >= self.thetas[7]):
            l_4_high = self.thetas[7]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)
        if (1.0 >= self.thetas[0]):
            l_9_high = self.thetas[0]
        else:
            l_9_high = 1.0
        if (0.0 >= l_9_high):
            l_10_lhs_integral = 0.0
        else:
            l_10_lhs_integral = (l_9_high - 0.0)
        l_6_cond = (1.0 - l_10_lhs_integral)

        return ((l_1_cond * (1.0 if (sample == []) else 0.0))
                + ((1.0 - l_1_cond)
                * ((l_6_cond * (0.0 if (sample == []) else
                    ((density_IRNormal((((sample)[0] - self.thetas[4]) / self.thetas[3])) / self.thetas[3])
                    * (0.0 if ((sample)[1:] == [])
                       else ((density_IRNormal(((((sample)[1:])[0] - self.thetas[6]) / self.thetas[5])) / self.thetas[5])
                          * self.forward(((sample)[1:])[1:]))))))
                    + ((1.0 - l_6_cond) * (0.0 if (sample == []) else ((density_IRNormal(
                               (((sample)[0] - self.thetas[2]) / self.thetas[1])) / self.thetas[1]) * self.forward((sample)[1:]))))
                    )))

    def generate(self):
        if (rand() >= self.thetas[7]):
            return []
        else:
            if (rand() >= self.thetas[0]):
                return [((randn() * self.thetas[3]) + self.thetas[4])] + [
                    ((randn() * self.thetas[5]) + self.thetas[6])] + self.generate()
            else:
                return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()


def converge_success_flattening(samples, search_params, epoch, batch_size):

    gausslist = Main_flattening()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(search_params))
    optimizer = optim.SGD(gausslist.parameters(), lr=0.01 / len(samples), momentum=0.001)
    criterion = torch.nn.NLLLoss()
    optimizer.zero_grad()

    for param in gausslist.parameters():
        param.data = param.data.double()
        if param.grad is not None:
            param.grad.data = param.grad.data.double()

    # Alternatively, you can use the following method to convert the whole model
    #gausslist = gausslist.double()

    indices_lists = []
    flattened_lists = []
    for i in range(int(len(samples) / batch_size)):
        indices = []
        flattened_list = []
        indices_even = []
        flattened_list_even = []
        indices_not = []
        indices_even_not = []
        indices_empty = []
        j = 0
        for sample in samples[batch_size * i: batch_size * i + batch_size]:
            if sample:
                indices += [j] * len(sample)
                flattened_list += sample
                if len(sample) == 1:
                    indices_even_not.append(j)
                if len(sample) > 1:
                    if len(sample) % 2 == 0:
                        indices_even += [j] * int((len(sample)) / 2)
                        flattened_list_even += sample
                    else:
                        indices_even_not.append(j)
            else:
                indices_not.append(j)
                indices_even_not.append(j)
            j += 1
        indices_lists.append([indices, indices_even, indices_not, indices_even_not])
        flattened_lists.append([flattened_list, flattened_list_even])

    guesses = []
    guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                    gausslist.thetas[3].item(), gausslist.thetas[4].item(), gausslist.thetas[5].item(),
                    gausslist.thetas[6].item(), gausslist.thetas[7].item()])
    execution = []
    execution.append(0)
    llh = []
    for epo in range(epoch):
        for i in range(int(len(samples)/batch_size)):
            start_time = time.time()
            likelihood = gausslist(batch_size, indices_lists[i], flattened_lists[i])
            #print("lk2: ", likelihood)
            #likelihood = -torch.log(gausslist(batch_size, indices_lists[i], flattened_lists[i]))
            end_time = time.time()
            execution.append((end_time - start_time) + execution[epo * int(len(samples)/batch_size) + i])
            """likelihood = torch.sum(likelihood)
            likelihood.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()"""
            llh += likelihood.tolist()
            guesses.append([gausslist.thetas[0].item(), gausslist.thetas[1].item(), gausslist.thetas[2].item(),
                            gausslist.thetas[3].item(), gausslist.thetas[4].item(), gausslist.thetas[5].item(),
                            gausslist.thetas[6].item(), gausslist.thetas[7].item()])

        print("Epoch report: ", epo)
        print("\t", gausslist.thetas)
    guesses = np.array(guesses)
    print(guesses)
    print(guesses.shape)
    return guesses, execution, llh

class Main_flattening(Module):
    def forward(self, batch, indices, flattened_list):
        if (1.0 >= self.thetas[7]):
            l_4_high = self.thetas[7]
        else:
            l_4_high = 1.0
        if (0.0 >= l_4_high):
            l_5_lhs_integral = 0.0
        else:
            l_5_lhs_integral = (l_4_high - 0.0)
        l_1_cond = (1.0 - l_5_lhs_integral)
        if (1.0 >= self.thetas[0]):
            l_9_high = self.thetas[0]
        else:
            l_9_high = 1.0
        if (0.0 >= l_9_high):
            l_10_lhs_integral = 0.0
        else:
            l_10_lhs_integral = (l_9_high - 0.0)
        l_6_cond = (1.0 - l_10_lhs_integral)
        #self.thetas = self.thetas.double()
        tmp1 = ((torch.ones(batch).double().index_reduce_(0, tensor(indices[1]).to(torch.int64),
                  ((1.0 - l_1_cond) * l_6_cond * ((density_IRNormal(((tensor(flattened_list[1]).double()[::2] - self.thetas[4]) / self.thetas[3])) / self.thetas[3])
                   * (density_IRNormal(((tensor(flattened_list[1]).double()[1::2] - self.thetas[6]) / self.thetas[5])) / self.thetas[5]))),
                  'prod') * l_1_cond)
                * (torch.ones(batch).double().index_reduce_(0, tensor(indices[3]).to(torch.int64),
                                            torch.zeros(len(indices[3])), 'amin')))

        tmp2 = (torch.ones(batch).double().index_reduce_(0, tensor(indices[0]).to(torch.int64),
                   ((1.0 - l_1_cond) * (1.0 - l_6_cond) * (density_IRNormal(((tensor(flattened_list[0]).double() - self.thetas[2]) / self.thetas[1])) / self.thetas[1])),
                 'prod') * l_1_cond)
                #* (torch.ones(batch).index_reduce_(0, tensor(indices[2]).to(torch.int64),
                 #                         torch.zeros(len(indices[2])), 'amin')))

        print("ind1: ", tensor(indices[1]).to(torch.int64))
        print("tmp1: ", tmp1)
        print("ind1: ", tensor(indices[0]).to(torch.int64))
        print("tmp2: ", tmp2)
        return (tmp1 + tmp2)



    def generate(self):
        if (rand() >= self.thetas[7]):
            return []
        else:
            if (rand() >= self.thetas[0]):
                return ([((randn() * self.thetas[3]) + self.thetas[4])]
                        + [((randn() * self.thetas[5]) + self.thetas[6])] + self.generate())
            else:
                return [((randn() * self.thetas[1]) + self.thetas[2])] + self.generate()



def exp_convergence():

    np.random.seed(22)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(21)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2], sample_params[3],
                     sample_params[4], sample_params[5], sample_params[6], sample_params[7]]
    gausslist = Program4()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for i in range(500):
        samples.append(gausslist.generate())
        #print("Sample: ", samples[i])

    guesses1, execution1 = converge_success(samples, search_params, 10)
    guesses2, execution2 = converge_success_flattening(samples, search_params, 10, 100)

    plt.plot(np.linspace(0, 10, 51), guesses1[:, 0], color="blue", label="Sequential BGD")
    plt.plot(np.linspace(0, 10, 251), guesses2[:, 0], color="orange", label="Flattening-parallelized MGD")
    plt.axhline(sample_params[0], color="gray", linestyle="dashed")
    plt.title("Generalization of flattening in SPLL")
    plt.xlabel("Training epoch")
    plt.ylabel(r"Learning parameter ($\theta_0$)")
    plt.legend()
    plt.xlim([0, 10])
    plt.ylim([0, 1])
    #plt.savefig("program3_convergence.png")
    plt.show()



def exp_llh_comparison():

    np.random.seed(10)
    sample_params = create_params()
    sample_params[0] = 0.8
    np.random.seed(21)
    search_params = create_params()
    search_params = [search_params[0], sample_params[1], sample_params[2], sample_params[3],
                     sample_params[4], sample_params[5], sample_params[6], sample_params[7]]
    gausslist = Program4()
    gausslist.thetas = torch.nn.parameter.Parameter(data=torch.tensor(sample_params))

    samples = []
    for i in range(10):
        samples.append(gausslist.generate())
        print("Sample: ", samples[i])

    guesses1, execution1, llh1 = converge_success(samples, search_params, 1)
    guesses2, execution2, llh2 = converge_success_flattening(samples, search_params, 1, 10)
    print("llh1: ", llh1)
    print("llh2: ", llh2)
    print("llh1: ", len(llh1))
    print("llh2: ", len(llh2))
    x = np.linspace(0, 10, 10)
    y = x
    plt.plot(x, y, zorder=1, color="gray", linestyle="dashed", label='Diagonal line')
    plt.scatter(np.array(llh1), np.array(llh2), zorder=1, color="blue")
    plt.title("Likelihood comparison of third program")
    plt.xlabel("Likelihood of sequential BGD")
    plt.ylabel("Likelihood of flattening-parallelized MGD")
    #plt.xlim([0, 8])
    #plt.ylim([0, 8])
    plt.grid(True)
    plt.legend()
    #plt.savefig("program3_llh_comparison.png")
    plt.show()


exp_convergence()
#exp_llh_comparison()