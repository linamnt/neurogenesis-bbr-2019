import torch
from torch.autograd import Variable
from torch.nn import Parameter
import torch.nn as nn
import torch.optim as optim
import numpy as np

from utils import *
from patterns import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    """
    Create the recurrent network.
    """
    def __init__(self, layer_size, weights):
        super(RNN, self).__init__()
        self.layer_size = layer_size
        # create weights
        self.r2r = nn.Linear(layer_size, layer_size)
        # replace the random weights with sparse weights
        self.r2r.weight.data = weights

    def forward(self, input_values, prev_state):
        combined = input_values + prev_state
        state = self.r2r(combined)
        return state

    def init_hidden(self, N):
        return Parameter(torch.zeros(N, self.layer_size)).to(device)


class Model(nn.Module):
    def __init__(self, layer_size, layer_sparsity, syn_sparsity,
                 mode='abba', steps=2):
        super(Model, self).__init__()
        '''
        Generate neural network.
        Input
        ===
        layer_size: list, [input, hidden, output] sizes
        layer_sparsity: list, [input, hidden, output] activation sparsities
        syn_sparsity: list, [input to hidden, hidden to output, rnn]
                      connectivity sparsities
        mode: str, 'abba' for reversal learning or 'abcd' for new learning
              in training_2
        steps: int > 0, number of steps the CA3 rnn takes.
        '''
        # parameters
        self.sig = nn.Sigmoid()
        self.dtype = torch.FloatTensor
        self.new = 0
        self.lr_new = 1
        self.steps = steps  # steps of the rnn

        self.layer_size = list(layer_size)
        self.layer_sparsity = list(layer_sparsity)
        self.layer_k = [int(i) for i in
                        layer_sparsity[1:] * np.array([layer_size[1],
                                                      layer_size[2]])]

        # a shell array for KWTA using topk()
        self.res_train1 = Variable(torch.zeros([n_train, layer_size[1]])).to(device)
        self.res_train2 = Variable(torch.zeros([n_train, layer_size[2]])).to(device)
        self.res_test1 = Variable(torch.zeros([n_test, layer_size[1]])).to(device)
        self.res_test2 = Variable(torch.zeros([n_test, layer_size[2]])).to(device)

        # get the number (for topk) of active neurons for each
        # layer based on the proportion
        self.syn_sparsity = list(syn_sparsity)
        self.syn_n = (syn_sparsity * np.array([layer_size[1], layer_size[2],
                                               layer_size[2]])).astype(int)
        for ix, i in enumerate(self.syn_sparsity):
            if i == 0:
                self.syn_sparsity[ix] = 1

        # initialize weights
        self.w1 = nn.Linear(layer_size[0], layer_size[1])
        self.w2 = nn.Linear(layer_size[1], layer_size[2])
        nn.init.xavier_uniform_(self.w1.weight)
        nn.init.xavier_uniform_(self.w2.weight))

        # recurrent weights are generated as a normal distribution scaled
        # by the synaptic sparsity of the recurrent layer
        self.wr = np.random.normal(loc=0, scale=np.sqrt(1./self.layer_size[1]),
                                   size=(layer_size[2], layer_size[2])
                                   )/np.sqrt(self.syn_sparsity[2])

        # need to convert to numpy as we can make sparse
        # connections using subset_zero()
        self.layers = [self.w1.weight.data.cpu().numpy(),
                       self.w2.weight.data.cpu().numpy(), self.wr]

        self.masks = []
        for ix, layer in enumerate(self.layers):
            for i in layer:
                subset_zero(i, self.syn_n[ix], 0)
            mask = np.array(layer)
            mask[mask != 0] = 1
            self.masks.append(mask)

        # get pre-generated dataset
        data = load_dataset(mode, valid=True)
        self.train = data[0]
        self.test = data[1]
        self.valid = data[2]
        self.n_train = len(self.train[0])
        self.n_test = len(self.test[0])

        # create the sparse masks
        self.w1_sp = Variable(torch.from_numpy(self.masks[0]).type(self.dtype),
                              requires_grad=False).to(device)
        self.w2_sp = Variable(torch.from_numpy(self.masks[1]).type(self.dtype),
                              requires_grad=False).to(device)
        self.wr_sp = torch.from_numpy(self.masks[2]).type(self.dtype).to(device)

        self.rnn = RNN(layer_size[2],
                       torch.from_numpy(self.layers[2]).type(self.dtype))

    def forward(self, x, shape):
        if self.new != 0:  # if there have been new neurons added
            l2 = torch.cat([self.sig(self.w1(x)), self.sig(self.w1_new(x))], 1)
            l2_old, l2_new = torch.split(l2, [self.layer_size[1]-self.new,
                                              self.new], dim=1)

            excite = int(self.excite*self.new)

            # excite requires a certain number of new neurons to be active
            # if no active new neurons, then continue as usual
            if (excite == 0) or (self.new_syn_n[0] == 0):
                l2_new = torch.zeros(l2_new.size()).to(device)
                # if no input connections but excite > 0
                excite = 0
            else:
                topk, indices = torch.topk(l2_new, excite)
                topk, indices = topk.to(device), indices.to(device)
                res = torch.zeros(l2_new.size()).to(device)
                l2_new = res.scatter(1, indices, topk)

            # if number of active new neurons is greater than proportion
            # of all neurons to be active,
            # then all mature neurons set to 0
            if excite > self.layer_k[0]:
                l2_old = torch.zeros(l2_old.size()).to(device)
            else:
                topk, indices = torch.topk(l2_old, self.layer_k[0])
                res = torch.zeros(l2_old.size()).to(device)
                l2_old = res.scatter(1, indices, topk)

            if shape == self.n_train:
                res = [self.res_train1, self.res_train2]
            else:
                res = [self.res_test1, self.res_test2]

            # get the input to CA3/RNN
            l3 = self.sig(self.w2(l2_old)) + self.sig(self.w2_new(l2_new))
            topk, indices = torch.topk(l3, self.layer_k[1])
            l3 = res[1].scatter(1, indices, topk)

        else:  # if no new neurons have been added
            if shape == self.n_train:
                res = [self.res_train1, self.res_train2]
                print(res[0].size())
            else:
                res = [self.res_test1, self.res_test2]

            l2 = self.sig(self.w1(x))
            topk, indices = torch.topk(l2, self.layer_k[0])
            topk, indices = topk.to(device), indices.to(device)
            l2 = res[0].scatter(1, indices, topk)

            l3 = self.sig(self.w2(l2))
            topk, indices = torch.topk(l3, self.layer_k[1])
            topk, indices = topk.to(device), indices.to(device)
            l3 = res[1].scatter(1, indices, topk)

        # ca3 recurrency
        state = self.rnn.init_hidden(shape)
        for i in range(self.steps):
            state = self.rnn(l3, state)
        topk, indices = torch.topk(state, self.layer_k[1])
        topk, indices = topk.to(device), indices.to(device)
        state = res[1].scatter(1, indices, topk)

        return state

    def update(self, epochs, optimizer, stage=0, log=False, generalize=False):
        """
        Train the network and update the weights using backpropagation.
        Input
        ===
        - epochs: int, number of updates
        - optimizer: torch.optim object, optimizer to update the weights
        - stage: int {0,1}, training stage, original patterns (0) or
                 new/conflicting patterns (1) default = 0, original patterns
        - log: bool, if True, keep track of performance after each update
        - generalize: bool, if True, log will keep track of the Test performance
        """
        # log the performance after each update
        if log:
            logger = np.zeros((epochs+1, 2))
            logger[0] = self.accuracy(generalize=generalize)

        assert stage in [0, 1]

        ntrain = len(self.train[0][stage])

        for epoch in range(epochs):
            self.training = True

            # Manually zero the gradients after updating weights
            optimizer.zero_grad()

            # get the predictions + loss
            output = self.forward(self.train[0][stage], self.n_train)
            loss = (output - self.train[1][stage]).abs().sum()
            loss.backward()

            # update the weights
            optimizer.step()
            # apply the sparse masks and clamp values between -1/1
            self.w1.weight.data *= self.w1_sp.data
            self.w1.weight.clamp(min=-1, max=1)
            self.w2.weight.data *= self.w2_sp.data
            self.w2.weight.clamp(min=-1, max=1)

            if self.new:
                self.w1_new.weight.data *= self.w1_new_sp.data
                self.w2_new.weight.data *= self.w2_new_sp.data
                self.w1_new.weight.clamp(min=-1, max=1)
                self.w2_new.weight.clamp(min=-1, max=1)

            for ix, p in enumerate(self.rnn.parameters()):
                if ix < 1:
                    # clip weights so CA3 doesn't explode using max normalization
                    p.data.mul_(self.wr_sp)
                    p.data = p.data.clamp(min=-1, max=1)
                if ix > 0:
                    pass

            if log:
                logger[epoch+1] = self.accuracy(generalize=generalize)
        if log:
            return logger

    def add_new(self, pnew=0.01, synio=[0.02, 0.005],
                excite=0.1, new_lr=1, replace=False):
        """
        Add new neurons to the network.

        Input
        ===
        - pnew: 0 < float < 1, proportion of the DG of which to
                add as new neurons
        - synio: [float, float], input and output connectivity sparsity
                 of new neurons
        - excite: 0 < float < 1, proportion of new neurons active
                  each forward pass
        - new_lr: float > 0, factor to convert mature to new learning rate
        - replace: bool, if True, replace old neurons as we add new neurons
        """
        # parameter updates
        if isinstance(pnew, int):
            n_new = pnew
        elif isinstance(pnew, float):
            n_new = int(self.layer_size[1]*pnew)
        if n_new == 0:
            return
        self.new_syn_sparsity = synio
        # number of synapses input/output for a new neuron
        self.new_syn_n = (synio*np.array([self.layer_size[0],
                                          self.layer_size[2]])
                          ).astype('int')
        self.new += n_new
        self.excite = excite
        self.lr_new = new_lr
        self.layer_size[1] += n_new

        # add new neurons to hidden layer
        # if new neurons have already been added before
        if self.new > n_new:
            current_new_wi = [self.w1_new.weight.data, self.w1_new.bias.data]
            current_new_wo = self.w2_new.weight.data

            newwi = torch.zeros([n_new, self.layer_size[0]])
            newwo = torch.zeros([self.layer_size[2], n_new])
            for i, wm in enumerate([newwi, newwo]):
                nn.init.xavier_uniform_(wm)
                wm = wm.clamp(min=-1, max=1)

            # must transpose first matrix
            for i in newwi:
                subset_zero(i, self.new_syn_n[0], 0)

            newwoT = newwo.transpose(0, 1)
            for i in newwoT:
                subset_zero(i, self.new_syn_n[1], 0)
            newwo = newwoT.transpose(0, 1)

            new_wi = torch.cat([current_new_wi[0], newwi], dim=0)
            new_wo = torch.cat([current_new_wo, newwo], dim=1)

            # update the masks
            for ix, i in enumerate([new_wi.numpy(), new_wo.numpy()]):
                mask = np.array(i)
                mask[mask != 0] = 1
                self.masks[ix] = mask
            self.w1_new_sp = Variable(torch.from_numpy(self.masks[0]).type(self.dtype),
                                      requires_grad=False)
            self.w2_new_sp = Variable(torch.from_numpy(self.masks[1]).type(self.dtype),
                                      requires_grad=False)

            self.w1_new = nn.Linear(self.layer_size[0], self.new)
            self.w2_new = nn.Linear(self.new, self.layer_size[2], bias=False)

            self.w1_new.weight.data = torch.tensor(new_wi, requires_grad=True)
            self.w2_new.weight.data = torch.tensor(new_wo, requires_grad=True)

            # maintain biases from before
            self.w1_new.bias.data[:self.new - n_new] = current_new_wi[1]
        else:
            self.w1_new = nn.Linear(self.layer_size[0], self.new, bias=True)
            self.w2_new = nn.Linear(self.new, self.layer_size[2], bias=False)

            # initialize new weights for new neurons
            nn.init.xavier_uniform_(self.w1_new.weight)
            nn.init.xavier_uniform_(self.w2_new.weight)

            # sparsity
            layers = [i.weight.data.cpu().numpy()
                      for i in [self.w1_new, self.w2_new]]
            for ix, layer in enumerate(layers):
                if ix == 1:
                    layer = layer.T
                for i in layer:
                    try:
                        subset_zero(i, self.new_syn_n[ix], 0)
                    except ValueError:
                        print(self.new_syn_n[ix], layer.shape, ix)
                if ix == 1:
                    layer = layer.T
                mask = np.array(layer)
                mask[mask != 0] = 1
                self.masks.append(mask)

            # update the masks
            self.w1_new_sp = Variable(torch.from_numpy(self.masks[-2]).type(self.dtype),
                                      requires_grad=False).to(device)
            self.w2_new_sp = Variable(torch.from_numpy(self.masks[-1]).type(self.dtype),
                                      requires_grad=False).to(device)

        # put the new place-holders on gpu if applicable
        self.res_train1 = Variable(torch.zeros([self.n_train,
                                                self.layer_size[1]])).to(device)
        self.res_test1 = Variable(torch.zeros([self.n_test,
                                               self.layer_size[1]])).to(device)
        if replace:
            n_lost = int(replace*n_new)
            self.remove_neurons(n_lost, self.new_syn_n[1], mode)

    def remove_neurons(self, neuron_num):
        """
        Remove mature neurons from the network.

        Input
        ===
        - neuron_num: int, number of neurons to remove
        """
        currentwi = self.w1.weight.data.clone()
        currentwo = self.w2.weight.data.clone()
        biaswi = self.w1.bias.data
        biaswo = self.w2.bias.data
        indices = np.random.choice(range(currentwo.shape[1]),
                                   neuron_num, replace=False)
        currentwi = np.delete(currentwi, indices, axis=0)
        currentwo = np.delete(currentwo, indices, axis=1)
        biaswi = np.delete(biaswi, indices, axis=0)

        # update the masks
        for ix, i in enumerate([currentwi.cpu().numpy(), currentwo.cpu().numpy()]):
            mask = np.array(i)
            mask[mask != 0] = 1
            self.masks[ix] = mask

        self.w1_sp = Variable(torch.from_numpy(self.masks[0]).type(self.dtype),
                              requires_grad=False).to(device)
        self.w2_sp = Variable(torch.from_numpy(self.masks[1]).type(self.dtype),
                              requires_grad=False).to(device)

        # update the weights
        self.layer_size[1] = currentwi.shape[0] + self.new
        self.w1 = nn.Linear(self.layer_size[0], self.layer_size[1] - self.new)
        self.w2 = nn.Linear(self.layer_size[1] - self.new, self.layer_size[2])

        # maintain the same weights from the preexisting neurons
        self.w1.weight.data = currentwi.clone().detach().requires_grad_(True)
        self.w2.weight.data = currentwo.clone().detach().requires_grad_(True)

        # maintain biases from before
        self.w1.bias.data = biaswi
        self.w2.bias.data = biaswo

        # maintain shapes for implementing topk
        self.res_train1 = Variable(torch.zeros([self.n_train,
                                                self.layer_size[1]])).to(device)
        self.res_test1 = Variable(torch.zeros([self.n_test,
                                               self.layer_size[1]])).to(device)


    def test_model(self, stage, generalize=True, valid=False, return_dg=False):
        """
        Get the hamming loss of the model.

        Input
        ===
        - stage: int [0,1], original patterns[0] or reversal[1] or new patterns[1]
        - generalize, bool, if True use test data, otherwise training data

        Output
        ===
        - hamming_loss: int, number of binary differences between the
                        target and predictions
        """
        if generalize:
            patterns = self.test
            N = self.n_test
            res = [self.res_test1, self.res_test2]
        elif valid:
            patterns = self.valid
            N = self.n_test
            res = [self.res_test1, self.res_test2]
        else:
            patterns = self.train
            N = self.n_train
            res = [self.res_train1, self.res_train2]

        output = self.forward(patterns[0][stage], N)
        indices = torch.nonzero(output, as_tuple=False)
        output = res[stage].scatter(1, indices, 1)
        l3_loss = (output - patterns[1][stage]).abs().sum()
        hamming_loss = l3_loss.data.cpu().numpy()/N

        return hamming_loss

    def accuracy(self, generalize=True, training_off=False, valid=False):
        '''
        Measure accuracy of network.

        Input
        ====
        - generalize: bool, if True, check test accuracy, otherwise training
        - training_off: bool, whether or not currently training the network
        - valid: bool, if True, check validation accuracy

        Output
        ===
        - stage_0: 0 < float < 100, accuracy of network on training_1
        - stage_1: 0 < float < 100, accuracy of network on training_2
        '''
        if training_off:
            self.training = False
        ca3_size = self.layer_k[1]
        stage_0 = (1 - self.test_model(0, generalize=generalize,
                                       valid=valid)/2/ca3_size)*100
        stage_1 = (1 - self.test_model(1, generalize=generalize,
                                       valid=valid)/2/ca3_size)*100
        return stage_0, stage_1
