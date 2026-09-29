from dataclasses import dataclass
import random

OPS = ('none','skip_connect','nor_conv_1x1','nor_conv_3x3','avg_pool_3x3')

@dataclass(frozen=True)
class Architecture:
    ops: tuple
    def __post_init__(self):
        if len(self.ops) != 6 or any(o not in OPS for o in self.ops):
            raise ValueError('TSS architecture must contain six valid operations.')
    @staticmethod
    def random(rng: random.Random):
        return Architecture(tuple(rng.choice(OPS) for _ in range(6)))
    def mutate(self, indices, rng):
        x=list(self.ops)
        for i in indices:
            choices=[o for o in OPS if o != x[i]]
            x[i]=rng.choice(choices)
        return Architecture(tuple(x))
    def distance(self, other):
        return sum(a != b for a,b in zip(self.ops, other.ops))/6.0
    def to_nats_string(self):
        # NAS-Bench-201/TSS edge order: (0->1),(0->2),(1->2),(0->3),(1->3),(2->3)
        o=self.ops
        return f'|{o[0]}~0|+|{o[1]}~0|{o[2]}~1|+|{o[3]}~0|{o[4]}~1|{o[5]}~2|'
