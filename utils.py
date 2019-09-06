import random


def subset_zero(array, synapses, setto=0):
    '''
    make sparse arrays based on desired number of synapses per row
    '''
    indices = random.sample(range(len(array)), len(array) - synapses)
    array[indices] = setto
