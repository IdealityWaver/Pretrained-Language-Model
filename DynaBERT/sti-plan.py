import sys

hw_prof = {'io':{}, 'comp':{}}

class shard():
    def __init__(self, n, m, b):
        self.n = n # which layer
        self.m = m # which vertical slice
        self.b = b # which bit fidelity

# to merge
class hw_prof():
    def __init__(self, prof):
        self.prof = prof

    def get_compute(self, num_shards):
        return self.prof['comp'][num_shards]
    
    def get_io(self, bits):
        return self.prof['io'][bits]

# below measured 
def init_hw_prof(hw_prof):
    # io delay of shards w/ diff. fidelity 
    hw_prof['io'][2] = 20 #ms
    hw_prof['io'][3] = 30
    hw_prof['io'][4] = 40
    hw_prof['io'][5] = 50
    hw_prof['io'][6] = 60

    # comp w/ diff # of shards per layer
    hw_prof['comp'][3] = 30 #ms
    hw_prof['comp'][4] = 40
    hw_prof['comp'][5] = 50
    hw_prof['comp'][6] = 60
    hw_prof['comp'][7] = 70
    hw_prof['comp'][8] = 80
    hw_prof['comp'][9] = 90
    hw_prof['comp'][10] = 100
    hw_prof['comp'][11] = 110
    hw_prof['comp'][12] = 120 
    return

def init_preload_shard():
    _s = set(shard(0,0,6), shard(0,1,6), shard(0, 2, 6), shard(0, 3, 6))
    return _s

def get_comp(hw_prof, num_shards):
    return hw_prof['comp'][num_shards]

def get_io(hw_prof, bits):
    return hw_prof['io'][bits]

def plan_compute(ddl, hw_prof):
    # n: layers 
    min_n = 6
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

def deduct_aib(n, t_io, aibs):
    for i in range(n, 12):
        aibs[i] = aibs[i] - t_io

def check_aibs(n, aibs):
    for i in range(0, n):
        if aibs[i] < 0:
            print("Constraints not satisfied!!")
            return 1 
    return 0

def plan_io(n, m, preload_buf, hw_prof, shard_prof):
    submodel = [[0] * m] * n
    print(submodel)
    # initialize AIBs 
    aibs = init_aibs(n, m, hw_prof, preload_buf)
    print(aibs)
    # fill submodel with preloaded shards
    for s in preload_buf:
        submodel[s.n][s.m] = s.b
        deduct_aib(n, get_io(hw_prof, s.b), aibs)
    # Pass 1: try to fill the rest of model with lowest-bit shard possible
    # Why 2-bit but not highest-bit possible? 
    # Because we want to provide enough room for important shards to have high fidelities
    for i in range(n):
        for j in range(m):
            if submodel[i][j] == 0:
                submodel[i][j] = 2
                deduct_aib(i, get_io(hw_prof, 2), aibs)
    check_aibs(n, aibs)
    # Pass 2: try to allocate high fidelities to important shards
    

    return

def plan(ddl, buf, hw_prof, shard_prof):
    # compute planning: get n x m submodel  
    (n, m) = plan_compute(ddl, hw_prof)
    print(n, m)
    # io planning: fill in the submodel w/ optimal fidelity
    conf = plan_io(n, m, buf, hw_prof, shard_prof)


init_hw_prof(hw_prof)
preload_shard = init_preload_shard()
plan(500, preload_shard, hw_prof, [])
