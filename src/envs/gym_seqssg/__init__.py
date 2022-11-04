import gym
import numpy as np

from .base import SeqSSG

# ============ Toy environments ============
n_rsc = 3
n_tgt = 5 
payoff_matrix = np.array([
    [1,   -1,   1.5, -1.2], 
    [0.9, -0.9, 0.7, -0.5],
    [0.8, -0.8, 0.5, -0.45],
    [1,   -1,   0.8, -0.7],
    [1.1, -1.1, 1.3, -1]
])
adj_matrix = np.array([
    [0,0,0,0,1],
    [0,0,1,1,0],
    [0,1,0,0,0],
    [0,1,0,0,1],
    [1,0,0,1,0]
])
cost_matrix = np.array([
    [ 0.  , -1.  , -1.  , -1.  ,  0.46],
    [-1.  ,  0.  ,  0.49,  0.38, -1.  ],
    [-1.  ,  0.49,  0.  , -1.  , -1.  ],
    [-1.  ,  0.38, -1.  ,  0.  ,  0.4 ],
    [ 0.46, -1.  , -1.  ,  0.4 ,  0.  ]
])
def_constraints = [(1,2)]
id_attr = {
    "SeqSSG-no-cstr-v0": {
        "has_constraint": False,
        "has_cost": False,
    }, 
    # Rationale for SeqSSG-v0:
    #   req_hop = 3: relax adjacency constraints
    #   req_dist = 1: enforce assignment constraints
    #   This makes sure that feasible actions exist and req_dist provides necessary
    #   constraints such that the problem would not be too easy.
    
    # "SeqSSG-v0": { 
    #     "has_constraint": True,
    #     "has_cost": True,
    #     "req_hop": 3,
    #     "req_dist": 1,
    # }
}

for env_id, attr in id_attr.items():
    kwargs = {
        "payoff_matrix": payoff_matrix,
        "adj_matrix": adj_matrix,
        "cost_matrix": cost_matrix,
        "def_constraints": def_constraints,
        "num_rsc": n_rsc,
        "num_tgt": n_tgt,
    }
    kwargs.update(attr)

    gym.register(id=env_id, entry_point=SeqSSG,
                max_episode_steps=100, kwargs=kwargs)
    # env = gym.make(env_id)

from gym import envs
# print("All registered envs")
# print(envs.registry.all())


