import numpy as np
import copy
import torch
from torch.autograd import Variable


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_rand01(neurons,sparsity):
    '''
    Chooses a random population of size n = neurons*sparsity
    from neurons.
    neurons: int, number of total units
    sparsity: 0 < float < 1, proportion of neurons to choose
    '''
    n = int(neurons * sparsity)
    if n == 0:
        n = 1
    k = neurons - n
    arr = np.array([0.] * k + [1.] * n)
    np.random.shuffle(arr)

    return arr


class patterns(object):
    def __init__(self, ni, no, clustered=True, overlap=0, sparsity=[0.05,0.1]):
        """
        ni: int, number of input neurons
        no: int, number of output neurons
        clustered: bool, whether the patterns are drawn from clusters of input
        overlap: 0 < float < 1, proportion of clusters overlapping
        sparsity: [float, float] where 0 < float < 1, a list of two values for
            input/output sparsity
        """
        self.ni = int(ni)
        self.no = int(no)
        self.clustered = clustered
        self.sparsity = sparsity
        self.overlap = int(ni*sparsity[0]*overlap)
        if clustered is not None:
            n_per_cluster = int(self.ni/4)
            neurons = np.arange(0,self.ni)
            # try removing shuffle if not learning TODO
            np.random.shuffle(neurons)
            self.cluster = np.array(np.split(neurons, 4))
            self.cluster_overlap = [
                np.ravel(self.cluster[1:4]),
                np.ravel(self.cluster[[0,2,3]]),
                np.ravel(self.cluster[[0,1,3]]),
                np.ravel(self.cluster[0:3])
                ]
            self.cluster_output = [make_rand01(no, self.sparsity[1]),
                        make_rand01(no, self.sparsity[1]), make_rand01(no, self.sparsity[1]),
                        make_rand01(no, self.sparsity[1])]


    def generate_pattern(self, cluster=None, generalize=True):
        '''
        Generates an input and output training pattern. If cluster is indicated,
        input will be drawn from that cluster group.
        '''

        if cluster is not None:
            input_pattern = np.zeros(self.ni)
            if generalize:
                size = int(self.sparsity[0] * self.ni)
                #choose which indices become 1
                input_indices = np.random.choice(self.cluster[cluster], size=size)
                input_pattern[input_indices] = 1
            else:
                size = int(self.sparsity[0] * self.ni - self.overlap)
                input_indices = np.random.choice(self.cluster[cluster], size=size)
                overlap_indices = np.random.choice(self.cluster_overlap[cluster], size=self.overlap)
                input_pattern[input_indices] = 1
                input_pattern[overlap_indices] = 1
            output_pattern = self.cluster_output[cluster]
        else:
            input_pattern = make_rand01(self.ni, self.sparsity[0])
            output_pattern = make_rand01(self.no, self.sparsity[1])

        return [input_pattern, output_pattern]


    def multi_patterns(self, npatterns, generalize=True):
        '''
        Generate npatterns patterns
        '''
        p = []
        if self.clustered:
            for i in range(npatterns):
                if i % 4 == 0:
                    p.append(self.generate_pattern(0, generalize=generalize))
                if i % 4 == 1:
                    p.append(self.generate_pattern(1, generalize=generalize))
                if i % 4 == 2:
                    p.append(self.generate_pattern(2, generalize=generalize))
                if i % 4 == 3:
                    p.append(self.generate_pattern(3, generalize=generalize))
        else:
            for i in range(npatterns):
                p.append(self.generate_pattern())
        return p

    def self_patterns(self, npatterns):
        '''
        Make patterns using self.multi_patterns, and set self.p to these patterns
        '''
        p = self.multi_patterns(npatterns)
        self.p = p


def make_patterns(n_in, n_out, n_train, n_test, sparsity, mode='abba', overlap=0.1):
    assert mode in ['abba', 'abcd'], "Mode must be one of ['abba', 'abcd']."
    pat = patterns(n_in, n_out, clustered=True, overlap=overlap, sparsity=sparsity)
    train = pat.multi_patterns(n_train*2, generalize=False)
    test = pat.multi_patterns(n_test*2, generalize=True)
    if mode == 'abcd':
        train_ab = [train[x] for x in range(len(train)) if x % 4 < 2]
        train_cd = [train[x] for x in range(len(train)) if x % 4 > 1]
        test_ab = [test[x] for x in range(len(test)) if x % 4 < 2]
        test_cd = [test[x] for x in range(len(test)) if x % 4 > 1]
        #0 = ab, 1 = cd
        train_in = [np.array([i[0] for i in train_ab]), np.array([i[0] for i in train_cd])]
        train_out = [np.array([i[1] for i in train_ab]), np.array([i[1] for i in train_cd])]
        test_in = [np.array([i[0] for i in test_ab]), np.array([i[0] for i in test_cd])]
        test_out = [np.array([i[1] for i in test_ab]), np.array([i[1] for i in test_cd])]
    elif mode == 'abba':
        train_ab = [train[x] for x in range(len(train)) if x % 4 < 2]
        test_ab = [test[x] for x in range(len(test)) if x % 4 < 2]
        train_ba = copy.deepcopy(train_ab)
        test_ba = copy.deepcopy(test_ab)
        train_cd = [train[x] for x in range(len(train)) if x % 4 > 1]
        test_cd = [test[x] for x in range(len(test)) if x % 4 > 1]

        for ix in range(len(train_ba)):
            if ix % 2 == 0:
                train_ba[ix] = [train_ba[ix][0], pat.cluster_output[1]]
            else:
                train_ba[ix] = [train_ba[ix][0], pat.cluster_output[0]]
        for ix in range(len(test_ba)):
            if ix % 2 == 0:
                test_ba[ix] = [test_ba[ix][0], pat.cluster_output[1]]
            else:
                test_ba[ix] = [test_ba[ix][0], pat.cluster_output[0]]

        #0 = ab, 1 = ba
        train_in = [np.array([i[0] for i in train_ab]), np.array([i[0] for i in train_ba]), np.array([i[0] for i in train_cd])]
        train_out = [np.array([i[1] for i in train_ab]), np.array([i[1] for i in train_ba]), np.array([i[1] for i in train_cd])]
        test_in = [np.array([i[0] for i in test_ab]), np.array([i[0] for i in test_ba]), np.array([i[0] for i in test_cd])]
        test_out = [np.array([i[1] for i in test_ab]), np.array([i[1] for i in test_ba]), np.array([i[1] for i in test_cd])]
    return train_in, train_out, test_in, test_out


def generate_dataset(layer_size, n_train, n_test, layer_sparsity, overlap):
    #make patterns
    dtype = torch.FloatTensor
    for mode in ['abba', 'abcd']:
        x_train, y_train, x_test, y_test = make_patterns(layer_size[0],
        layer_size[2], n_train, n_test, [layer_sparsity[0], layer_sparsity[2]], mode, overlap)
        #each of these will be a list, whereby index 0 = stage 1, index 1 = stage 2
        x_train = [Variable(torch.from_numpy(i).type(dtype),
                            requires_grad=False).to(device) for i in x_train]
        y_train = [Variable(torch.from_numpy(i).type(dtype),
                            requires_grad=False).to(device) for i in y_train]
        x_test = [Variable(torch.from_numpy(i).type(dtype),
                            requires_grad=False).to(device) for i in x_test]
        y_test = [Variable(torch.from_numpy(i).type(dtype),
                            requires_grad=False).to(device) for i in y_test]

        _, _, x_valid, y_valid = make_patterns(layer_size[0],
        layer_size[2], n_train, n_test, [layer_sparsity[0], layer_sparsity[2]], mode, overlap)

        x_valid = [Variable(torch.from_numpy(i).type(dtype),
                            requires_grad=False).to(device) for i in x_valid]
        y_valid = [Variable(torch.from_numpy(i).type(dtype),
                            requires_grad=False).to(device) for i in y_valid]


        train = [x_train, y_train]
        test = [x_test, y_test]
        valid = [x_valid, y_valid]

        torch.save(train, 'train_{}.pt'.format(mode))
        torch.save(valid, 'valid_{}.pt'.format(mode))
        torch.save(test, 'test_{}.pt'.format(mode))

        print("Saved files.")




def load_dataset(mode, valid=False):
    train = torch.load('train_{}.pt'.format(mode))
    test = torch.load('test_{}.pt'.format(mode))
    if valid:
        valid = torch.load('valid_{}.pt'.format(mode))

        return train, test, valid
    else:
        return train, test