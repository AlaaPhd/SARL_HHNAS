from benchmark.architecture import Architecture

def h1(a,best,rng): return a.mutate([rng.randrange(6)],rng)
def h2(a,best,rng): return a.mutate(rng.sample(range(6),2),rng)
def h3(a,best,rng):
    pairs=[(i,j) for i in range(6) for j in range(i+1,6) if a.ops[i]!=a.ops[j]]
    if not pairs: return h1(a,best,rng)
    i,j=rng.choice(pairs); x=list(a.ops); x[i],x[j]=x[j],x[i]
    return Architecture(tuple(x))
def h4(a,best,rng):
    diff=[i for i in range(6) if a.ops[i]!=best.ops[i]]
    if not diff: return h1(a,best,rng)
    i=rng.choice(diff); x=list(a.ops); x[i]=best.ops[i]
    return Architecture(tuple(x))
def h5(a,best,rng): return a.mutate(rng.sample(range(6),3),rng)
def h6(a,best,rng):
    for _ in range(100):
        c=Architecture.random(rng)
        if c.distance(best)>=0.5: return c
    return a.mutate(rng.sample(range(6),3),rng)

LLHS=(h1,h2,h3,h4,h5,h6)
LLH_NAMES=('H1_single','H2_double','H3_swap','H4_guided','H5_triple','H6_restart')
