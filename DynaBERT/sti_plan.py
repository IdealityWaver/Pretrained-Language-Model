import sys
import numpy as np


class shard():
    def __init__(self, n, m, b):
        self.n = n # which layer
        self.m = m # which vertical slice
        self.b = b # which bit fidelity

# to merge
'''
class hw_prof():
    def __init__(self, prof):
        self.prof = prof

    def get_compute(self, num_shards):
        return self.prof['comp'][num_shards]
    
    def get_io(self, bits):
        return self.prof['io'][bits]
'''

# below measured 
def init_hw_prof():
    hw_prof = {'io':{}, 'comp':{}}
    # io delay of one shard w/ diff. fidelity 
    layer_delay = 339.31 #ms
    shard_delay = layer_delay / 12
    for bit in range(2, 33):
        quantized_delay = shard_delay * bit / 32
        hw_prof['io'][bit] = quantized_delay
    print(hw_prof['io'])

    # comp delay w/ diff # of shards per layer, including decompression
    decomp_shard = 2
    hw_prof['comp'][3] = 35.0 #ms 
    hw_prof['comp'][4] = 44.0 
    hw_prof['comp'][5] = 52.0 
    hw_prof['comp'][6] = 58.0 
    hw_prof['comp'][7] = 65.0 
    hw_prof['comp'][8] = 70.0
    hw_prof['comp'][9] = 75.0
    hw_prof['comp'][10] = 83.0
    hw_prof['comp'][11] = 87.0
    hw_prof['comp'][12] = 95.0 
    for num in range(3, 13):
        hw_prof['comp'][num] = hw_prof['comp'][num] + num * decomp_shard
    return hw_prof

# m 6-bit preload shards enough for one layer
def init_preload_shard(m):
    s = set()
    for i in range(m): 
        _s = shard(0, i, 6)
        s.add(_s)
    return s

def get_comp(hw_prof, num_shards):
    return hw_prof['comp'][num_shards]

def get_io(hw_prof, bits):
    return hw_prof['io'][bits]

def plan_compute(ddl, hw_prof):
    # n: layers 
    min_n = 3
    max_n = 12
    # m: shards
    min_m = 3
    max_m = 12
    # we want to a model larger than 6x3
    min_ddl = min_n * hw_prof['comp'][min_m] 
    if ddl < min_ddl:
        print("DDL too small... ( < %f ms )" % (min_ddl))
        return (min_n, min_m)
    #  find largest possible submodel, bound by compute
    for n in range(max_n, min_n - 1, -1):
        for m in range(max_m, min_m -1, -1):
            comp = get_comp(hw_prof, m)
            if n * comp > ddl:
                continue
            else:
                print("submodel: (%d, %d)" % (n, m))
                return (n, m)
    return (min_n, min_m)
    print("Cannot find a valid n, m!! Check DDL...")

def init_aibs(n, m, hw_prof, buf):
    aibs = [0] * n
    free_io = 0
    comp = get_comp(hw_prof, m)
    for shard in buf:
        free_io = free_io + get_io(hw_prof, shard.b)
    for i in range(0, n):
        aibs[i] = i * comp + free_io
    return aibs

def deduct_aib(curr, total_layers, t_io, aibs):
    for i in range(curr, total_layers):
        aibs[i] = aibs[i] - t_io
        if aibs[i] < 0:
            print("Cannot deduct... aib[%d] negative!!" % (i))

def can_deduct_aib(curr, total_layers, t_io, aibs):
    for i in range(curr, total_layers):
        tmp = aibs[i] - t_io
        if tmp < 0:
            return 0
            print("Cannot deduct... aib[%d] negative!!" % (i))
    return 1

def check_aibs(n, aibs):
    for i in range(0, n):
        if aibs[i] < 0:
            print("Constraints not satisfied!!")
            return 1 
    return 0

def plan_io(n, m, preload_buf, hw_prof, shard_prof):
    submodel = np.zeros(shape=(n, m))
    # initialize AIBs 
    aibs = init_aibs(n, m, hw_prof, preload_buf)
    print(aibs)
    # fill submodel with preloaded shards
    for s in preload_buf:
        submodel[s.n, s.m] = s.b
        deduct_aib(s.n, n, get_io(hw_prof, s.b), aibs)
    print(submodel)
    # Pass 1: try to fill the rest of model with lowest-bit shard possible
    # Why 2-bit but not highest-bit possible? 
    # Because we want to reserve enough room for important shards to have high fidelities
    for i in range(n):
        for j in range(m):
            if submodel[i, j] == 0:
                submodel[i, j] = 2
                deduct_aib(i, n, get_io(hw_prof, 2), aibs)
    check_aibs(n, aibs)
    print("base, after loading preload buffer...")
    print(submodel)
    # Pass 2: try to allocate 6-bit fidelities to important shards
    target_fidel = 6
    ranked_importance = np.dstack(np.unravel_index(np.argsort(shard_prof.ravel()), (12, 12)))[0][::-1]
    for s in ranked_importance:
        # print(shard_prof[s[0], s[1]])
        s_n = s[0]
        s_m = s[1]
        # in the submodel we care about
        if s_n < n and s_m < m:
            old_fidel = submodel[s_n, s_m]
            if old_fidel == 2:
                delta = target_fidel - old_fidel
                deduct_aib(s_n, n, get_io(hw_prof, delta), aibs)
                submodel[s_n, s_m] = target_fidel
                if check_aibs(n, aibs) == 1:
                    print("AIBs used up, return submodel")
                    return submodel
    # Pass 3: still has some AIB to spare. Try 32 bit
    target_fidel = 32
    max_fidel = int(aibs[n-1]/get_io(hw_prof, 32-6))
    print("AIBs not used up (last AIB=%f), can upgrade %d to 32bit" % (aibs[n-1], max_fidel))
    for s in ranked_importance:
        s_n = s[0]
        s_m = s[1]
        allocated = 0
        # in the submodel we care about
        if s_n < n and s_m < m and allocated < max_fidel:
            old_fidel = submodel[s_n, s_m]
            delta = target_fidel - old_fidel
            if can_deduct_aib(s_n, n, get_io(hw_prof, delta), aibs):
                allocated += 1
                deduct_aib(s_n, n, get_io(hw_prof, delta), aibs)
                submodel[s_n, s_m] = target_fidel
                if check_aibs(n, aibs) == 1:
                    print("AIBs used up, return submodel")
                    return submodel
    return submodel

def read_shard_importance(path):
    import re
    pattern = ".*: (0.*)\}: \(([0-9]*),([0-9]*)\)"
    pattern = re.compile(pattern)
    prof = np.zeros(shape=(12, 12))
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            res = pattern.match(line)
            if res == None:
                continue
            acc, m, n = float(res[1]), int(res[2]), int(res[3])
            prof[n, m] = acc
    print("Successfully read in shard importance profile...")
    return prof



def _plan(ddl, buf, hw_prof, shard_prof, n, m):
    # compute planning: get n x m submodel  
    if n == 0 and m == 0:
        (n, m) = plan_compute(ddl, hw_prof)
        print("compute planning: n = %d, m = %d" % (n, m))
    else:
        print("using supplied n = %d, m = %d" % (n, m))
    # io planning: fill in the submodel w/ optimal fidelity
    conf = plan_io(n, m, buf, hw_prof, shard_prof)
    return conf


def plan(ddl, task, n=0, m=0):
    shard_prof = read_shard_importance('../../shard_importance/{0}_prof.txt'.format(task))
    hw_prof = init_hw_prof()
    preload_shard = init_preload_shard(m)
    submodel = _plan(ddl, preload_shard, hw_prof, shard_prof, n, m)
    print("final model:")
    a = "%s" % (str(submodel))
    print(a)
    print(submodel)
    return submodel

plan(300, 'qqp', 8, 3)
