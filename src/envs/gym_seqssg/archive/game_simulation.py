import itertools
import math
from collections.abc import Iterable

import networkx as nx
import numpy as np
import torch

from . import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GameSimulation(object):
    @staticmethod
    def gen_next_state(
        state: np.ndarray, 
        def_action: np.ndarray, 
        att_action: np.ndarray, 
        payoff_matrix: np.ndarray, 
        adj_matrix: np.ndarray, 
        keep_res=False
    ):
        # Defender action is a txt matrix showing resource movement
        num_target = payoff_matrix.shape[0]
        next_state = state.copy()
        def_immediate_utility = 0.0
        att_immediate_utility = 0.0

        next_state[:, 0] = def_action.sum(axis=0)

        protected = False
        for t in range(num_target):
            if att_action[t] == 1:
                next_state[t, 1] = 1                # target t is attacked
                if not keep_res:
                    if next_state[t, 0] <= 1:
                        next_state[t, 0] = 0                # one resource at t is destroyed
                    else:
                        next_state[t, 0] -= 1
                for tprime in range(num_target):
                    if def_action[tprime, t] >= 1:
                        def_immediate_utility = payoff_matrix[t, 0].copy()
                        att_immediate_utility = payoff_matrix[t, 3].copy()
                        protected = True
                        break
                if not protected:
                    def_immediate_utility = payoff_matrix[t, 1].copy()
                    att_immediate_utility = payoff_matrix[t, 2].copy()
                    break

        # Defender resource movement cost
        not_start = state[:, 1].sum()
        for t in range(num_target):
            for tprime in range(num_target):
                if def_action[tprime, t] >= 1 and not_start:
                    def_immediate_utility += adj_matrix[tprime, t] * def_action[tprime, t]

        return next_state, def_immediate_utility, att_immediate_utility

    @staticmethod
    def _gen_next_state(state, def_action, att_action, payoff_matrix, adj_matrix, keep_res=0):
        # Defender action is a txt matrix showing resource movement
        num_target = payoff_matrix.size(0)
        next_state = state.clone()
        def_immediate_utility = 0.0
        att_immediate_utility = 0.0

        next_state[:, 0] = def_action.sum(dim=0)

        protected = False
        for t in range(num_target):
            if att_action[t] == 1:
                next_state[t, 1] = 1                # target t is attacked
                if not keep_res:
                    if next_state[t, 0] <= 1:
                        next_state[t, 0] = 0                # one resource at t is destroyed
                    else:
                        next_state[t, 0] -= 1
                for tprime in range(num_target):
                    if def_action[tprime, t] >= 1:
                        def_immediate_utility = payoff_matrix[t, 0].clone()
                        att_immediate_utility = payoff_matrix[t, 3].clone()
                        protected = True
                        break
                if not protected:
                    def_immediate_utility = payoff_matrix[t, 1].clone()
                    att_immediate_utility = payoff_matrix[t, 2].clone()
                    break

        # Defender resource movement cost
        not_start = state[:, 1].sum()
        for t in range(num_target):
            for tprime in range(num_target):
                if def_action[tprime, t] >= 1 and not_start:
                    def_immediate_utility += adj_matrix[tprime, t] * def_action[tprime, t]

        return next_state, def_immediate_utility, att_immediate_utility

    @staticmethod
    def gen_next_observation(observation, def_action, att_action):
        next_observation = observation.clone()
        attack_idx = torch.nonzero(att_action).squeeze(1)  # .item()
        for idx in attack_idx:
            next_observation[idx, 1] = 1
            next_observation[idx, 0] = def_action[:, idx].sum()     # attacker only observes defender resources where it attacks
        return next_observation

    @staticmethod
    def sample_pure_strategy(mixed_strategy):
        idx = 0
        pivot = torch.rand(1)
        init_prob = mixed_strategy[0].probability
        while pivot > init_prob:
            idx += 1
            if idx >= len(mixed_strategy):
                idx -= 1
                break
            init_prob += mixed_strategy[idx].probability

        return mixed_strategy[idx]

    @staticmethod
    def sample_att_action_from_distribution(distribution, num_att, device):  # distribution is of size num_target
        num_target = distribution.size(0)
        non_zero_prob = torch.nonzero(distribution > 1e-4).squeeze(1)
        idx = 0
        count = 1  # make count = num_att and config.NUM_STEP = 1 for multiple attacks per time step
        att_action = torch.zeros(num_target, dtype=torch.float32, device=device)
        for _ in range(count):
            pivot = torch.rand(1)
            init_prob = distribution[non_zero_prob[0]].clone()
            while pivot > init_prob:
                idx += 1
                if idx >= len(non_zero_prob):
                    idx -= 1
                    break
                init_prob += distribution[non_zero_prob[idx]]
            att_action[non_zero_prob[idx].item()] = 1

        return att_action

    @staticmethod
    def sample_def_action_from_distribution(state, distributions, def_constraints, device):
        # Input: txt matrix for marginal distribution -- Output: txt binary matrix for resource movement
        num_target = state.size(0)
        def_current_location = torch.nonzero(state[:, 0])
        valid_action = False
        while not valid_action:
            def_action = torch.zeros(num_target, num_target).to(device)
            for t in def_current_location:
                probs = distributions[t.item()]
                non_zero_prob = torch.nonzero(probs > 1e-4).squeeze(1)
                count = state[t, 0].item()
                for _ in range(count):
                    pivot = torch.rand(1)
                    init_prob = probs[non_zero_prob[0]].clone()
                    idx = 0
                    while pivot > init_prob:
                        idx += 1
                        if idx >= len(non_zero_prob):
                            idx -= 1
                            break
                        init_prob += probs[non_zero_prob[idx]]
                    def_action[t, non_zero_prob[idx]] += 1
            # valid_action = check_def_constraints(def_action, def_constraints, state)  # for implementing defender constraints
            valid_action = True
            if not valid_action:
                print("Invalid Defender Move.")
        return def_action
    
    @staticmethod
    def sample_def_action_full(distributions, all_moves):
        # Samples defender action from full action space
        def_action = torch.zeros(config.NUM_RESOURCE, config.NUM_TARGET).to(device)
        idx = 0
        prob = distributions[idx]
        pivot = torch.rand(1)
        while pivot > prob:
            idx += 1
            prob += distributions[idx]
        def_code = all_moves[idx]

        for i,pos in enumerate(def_code):
            def_action[i][pos] = 1

        return def_action, distributions[idx]

    @staticmethod
    def sample_def_action_A2C(num_attack_remain, state, trained_strategy, def_constraints, device):
        num_target = state.size(0)
        if num_attack_remain < config.NUM_ATTACK and state[:, 0].sum() == 0:
            return torch.zeros(num_target, num_target, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                actor, critic = trained_strategy(state.unsqueeze(0))
            return GameSimulation.sample_def_action_from_distribution(state, actor.squeeze(0), def_constraints, device)

    @staticmethod
    def sample_def_action_A2C_LSTM(num_attack_remain, state, trained_strategy,
                                   action_hidden_state, action_cell_state, value_hidden_state, value_cell_state,
                                   def_constraints, device):
        num_target = state.size(0)
        if num_attack_remain < config.NUM_ATTACK and state[:, 0].sum() == 0:
            return torch.zeros(num_target, num_target, dtype=torch.float32, device=device), [], [], [], []
        else:
            with torch.no_grad():
                actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                    = trained_strategy(state.unsqueeze(0),
                                       action_hidden_state, action_cell_state, value_hidden_state, value_cell_state)
                def_action = GameSimulation.sample_def_action_from_distribution(state, actor.squeeze(0),
                                                                                def_constraints, device)
                return def_action, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state

    @staticmethod
    def sample_def_action_GAN(step, state, trained_strategy, adj, device):
        num_target = state.size(0)
        if step > 0 and state[:, 0].sum() == 0:
            return torch.zeros(num_target, num_target, dtype=torch.float32, device=device)
        else:
            with torch.no_grad():
                actor, critic = trained_strategy(state.unsqueeze(0))
            return GameSimulation.sample_def_action_from_res_dist(state, actor.squeeze(0), adj, device)

    @staticmethod
    def sample_def_action_uniform(state, adj_matrix, device):
        num_target = adj_matrix.size(0)
        def_action = torch.zeros(num_target, num_target).to(device)
        def_location = torch.nonzero(state[:, 0])
        temp_adj_matrix = torch.where(adj_matrix == config.MIN_VALUE, 0, 1)
        for target in def_location:
            num_defender = state[target.item(), 0].item()
            neighbors = torch.nonzero(temp_adj_matrix[target.item(), :]).squeeze(1)
            for _ in range(num_defender):
                idx = torch.randint(0, len(neighbors), (1,)).item()
                new_target = neighbors[idx]
                def_action[target.item(), new_target.item()] += 1

        return def_action

    @staticmethod
    def sample_att_action_uniform(state, device):
        num_target = state.size(0)
        not_attacked = torch.where(state[:, 1] < 1)[0]
        target = not_attacked[torch.randint(0, len(not_attacked), [1, ]).item()].item()
        attack = torch.zeros([num_target], dtype=torch.float32, device=device)
        attack[target] = 1
        return attack

    @staticmethod
    def sample_att_action_A2C(observation, trained_strategy, num_att, device):
        with torch.no_grad():
            actor, critic = trained_strategy(observation.unsqueeze(0))
        return GameSimulation.sample_att_action_from_distribution(actor.squeeze(0), num_att, device)

    @staticmethod
    def sample_att_action_A2C_LSTM(observation, trained_strategy,
                                   action_hidden_state, action_cell_state, value_hidden_state, value_cell_state,
                                   num_att, device):
        with torch.no_grad():
            actor, critic, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state \
                = trained_strategy(observation.unsqueeze(0), action_hidden_state,
                                   action_cell_state, value_hidden_state, value_cell_state)
            att_action = GameSimulation.sample_att_action_from_distribution(actor.squeeze(0), num_att, device)
            return att_action, action_hidden_state, action_cell_state, value_hidden_state, value_cell_state

    @staticmethod
    def sample_def_action_suqr(state, payoff, adj, device):
        num_target = config.NUM_TARGET
        def_action = torch.zeros(num_target, num_target).to(device)
        def_location = torch.nonzero(state[:, 0])
        temp_adj_matrix = torch.where(adj == config.MIN_VALUE, 0, 1)

        for target in def_location:
            num_defender = state[target.item(), 0].item()
            neighbors = torch.nonzero(temp_adj_matrix[target.item(), :]).squeeze(1)
            available_pay = [p[:2] for i,p in enumerate(payoff) if i in neighbors]
            def_qr = get_util_probs(available_pay)
            for _ in range(num_defender):
                init_prob = def_qr[0]
                pivot = torch.rand(1)
                idx = 0
                while pivot > init_prob:
                    idx += 1
                    if idx >= len(def_qr):
                        idx -= 1
                        break
                    init_prob += def_qr[idx]
                new_target = neighbors[idx]
                def_action[target.item(), new_target.item()] += 1

        return def_action

    @staticmethod
    def sample_att_action_suqr(
        state: np.ndarray, 
        payoff: np.ndarray
    ) -> np.ndarray:
        num_target = config.NUM_TARGET
        attack = np.zeros([num_target], dtype=np.float32)
        not_attacked = np.where(state[:, 1] < 1)[0]
        available_pay = [p[2:] for i, p in enumerate(payoff) if i in not_attacked]
        atk_qr = get_util_probs(available_pay)

        init_prob = atk_qr[0]
        pivot = np.random.uniform()
        idx = 0
        while pivot > init_prob:
            idx += 1
            if idx >= len(atk_qr):
                idx -= 1
                break
            init_prob += atk_qr[idx]

        attack[idx] = 1
        return attack

    @staticmethod
    def _sample_att_action_suqr(state, payoff, device=device):
        num_target = config.NUM_TARGET
        attack = torch.zeros([num_target], dtype=torch.float32, device=device)
        not_attacked = torch.where(state[:, 1] < 1)[0]
        available_pay = [p[2:] for i, p in enumerate(payoff) if i in not_attacked]
        atk_qr = _get_util_probs(available_pay, 'atk')

        init_prob = atk_qr[0]
        pivot = torch.rand(1)
        idx = 0
        while pivot > init_prob:
            idx += 1
            if idx >= len(atk_qr):
                idx -= 1
                break
            init_prob += atk_qr[idx]

        attack[idx] = 1
        return attack

    @staticmethod
    def sample_def_action_from_res_dist(state, distributions, def_constraints, adj, device, nc=0):
        # Returns Defender action as a (k x t) matrix where k is the number of resources and t is the number of targets
        # Input is (txt) probability matrix
        num_target = config.NUM_TARGET
        num_resource = config.NUM_RESOURCE
        def_location = torch.nonzero(state[:, 0])

        def_action = torch.zeros(num_resource, num_target).to(device)
        i = 0
        for target in def_location:
            probs = distributions[target][0].tolist()
            num_defender = state[target.item(), 0].item()
            for _ in range(num_defender):
                available_loc = [j for j in range(num_target)]
                constraint = [c for c in def_constraints if i in c][0]
                other_res = [r for r in constraint if r != i and sum(def_action[r]) > 0]
                if len(other_res) > 0:
                    for r in other_res:
                        r_loc = torch.nonzero(def_action[r])[0].item()
                        r_neighbors = torch.where(adj[r_loc]<config.MIN_VALUE)[0].tolist()
                        available_loc = [l for l in available_loc if l in r_neighbors]
                    temp_probs = [p for i,p in enumerate(probs) if i in available_loc]
                    probs = [p/sum(temp_probs) for p in temp_probs if sum(temp_probs) > 0.0]
                    if len(probs) < 1:
                        probs = distributions[target][0].tolist()
                        available_loc = [i for i in range(num_target)]
                pivot = torch.rand(1).to(device)
                init_prob = probs[0]
                idx = 0
                while pivot > init_prob:
                    idx += 1
                    if idx >= len(probs):
                        idx = probs.index(max(probs)) # torch.nonzero(probs)[-1].item()
                        break
                    init_prob += probs[idx]
                targ = available_loc[idx]
                def_action[i, targ] = 1
                i += 1

        return def_action

    @staticmethod
    def gen_next_state_from_def_res(state, def_action, att_action, def_cur_loc, payoff_matrix, adj_matrix):
        # Based on kxt matrix for defender input
        num_target = config.NUM_TARGET
        num_resource = config.NUM_RESOURCE
        next_state = state.clone()
        def_immediate_utility = 0.0
        att_immediate_utility = 0.0

        next_state[:, 0] = def_action.sum(dim=0)

        protected = False
        for t in range(num_target):
            if att_action[t] == 1:
                next_state[t, 1] = 1
                # next_state[t, 0] -= 1                 # defender resource not destroyed if it meets an attack
                for tprime in range(num_resource):
                    if def_action[tprime, t] >= 1:
                        def_immediate_utility = payoff_matrix[t, 0].clone()
                        att_immediate_utility = payoff_matrix[t, 3].clone()
                        protected = True
                        break
                if not protected:
                    def_immediate_utility = payoff_matrix[t, 1].clone()
                    att_immediate_utility = payoff_matrix[t, 2].clone()
                break
        '''
        # Calculating Defender movement cost
        not_start = state[:, 1].sum()
        for r in range(num_resource):
            for t in range(num_target):
                if def_action[r, t] >= 1 and def_cur_loc[r, t] < 1 and not_start:
                    prev_loc = torch.nonzero(def_cur_loc[r])[0].item()
                    # def_immediate_utility += adj_matrix[prev_loc, t]
        '''

        return next_state, def_immediate_utility, att_immediate_utility
    
    @staticmethod
    def play_game(def_strat, att_strat, payoff_matrix, adj_matrix, def_constraints, reach, d_option, a_option, keep_res=0):
        def_utility_average = 0.0
        att_utility_average = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_target = payoff_matrix.size(0)
        lstm_hidden_size = config.LSTM_HIDDEN_SIZE
        n_sample = 50

        for i_sample in range(n_sample):
            init_state = torch.zeros(num_target, 2, dtype=torch.int32, device=device)
            if 'GAN' in d_option:
                def_init_loc = gen_init_def_pos(num_target, config.NUM_RESOURCE, def_constraints, reach)
                for t, res in enumerate(def_init_loc):
                    init_state[(res == 1).nonzero(), 0] += int(sum(res))
            else:
                entries = torch.randint(0, num_target, [config.NUM_RESOURCE, ])
                for t in range(0, len(entries)):
                    init_state[entries[t], 0] += 1

            state = init_state
            init_attacker_observation = torch.zeros(num_target, 2, dtype=torch.int32, device=device)
            init_attacker_observation[:, 0] = -1
            attacker_observation = init_attacker_observation
            num_att = config.NUM_ATTACK

            if 'LSTM' in d_option:
                d_action_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                d_action_cell_state = torch.zeros(1, lstm_hidden_size, device=device)
                d_value_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                d_value_cell_state = torch.zeros(1, lstm_hidden_size, device=device)

            if 'LSTM' in a_option:
                a_action_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                a_action_cell_state = torch.zeros(1, lstm_hidden_size, device=device)
                a_value_hidden_state = torch.zeros(1, lstm_hidden_size, device=device)
                a_value_cell_state = torch.zeros(1, lstm_hidden_size, device=device)

            # for t in range(config.NUM_STEP):
            while num_att > 0:
                with torch.no_grad():
                    if 'LSTM' in d_option:
                        def_actor, def_critic, d_action_hidden_state, d_action_cell_state, d_value_hidden_state, d_value_cell_state \
                            = def_strat(state=state.unsqueeze(0), action_hidden_state=d_action_hidden_state,
                                        action_cell_state=d_action_cell_state, value_hidden_state=d_value_hidden_state,
                                        value_cell_state=d_value_cell_state)
                    else:
                        def_actor, def_critic = def_strat(state=state.unsqueeze(0))

                    if 'LSTM' in a_option:
                        att_actor, att_critic, a_action_hidden_state, a_action_cell_state, a_value_hidden_state, a_value_cell_state \
                            = att_strat(state=attacker_observation.unsqueeze(0),
                                        action_hidden_state=a_action_hidden_state,
                                        action_cell_state=a_action_cell_state, value_hidden_state=a_value_hidden_state,
                                        value_cell_state=a_value_cell_state)
                    else:
                        att_actor, att_critic = att_strat(state=attacker_observation.unsqueeze(0))

                    if num_att < config.NUM_ATTACK and state[:, 0].sum() == 0:
                        def_action = torch.zeros(num_target, num_target, dtype=torch.float32, device=device)
                    elif 'GAN' in d_option:
                        def_action = GameSimulation.sample_def_action_from_res_dist(state=state, distributions=def_actor.squeeze(0),
                                                                                    device=device)
                    else:
                        def_action = GameSimulation.sample_def_action_from_distribution(state=state, distributions=def_actor.squeeze(0),
                                                                                        def_constraints=def_constraints,
                                                                                        device=device)
                    att_action = GameSimulation.sample_att_action_from_distribution(distribution=att_actor.squeeze(0),
                                                                                    num_att=num_att,
                                                                                    device=device)
                    if 'GAN' in d_option:
                        next_state, def_immediate_utility, att_immediate_utility \
                            = GameSimulation.gen_next_state_from_def_res(state, def_action, att_action, payoff_matrix,
                                                                         adj_matrix)
                    else:
                        next_state, def_immediate_utility, att_immediate_utility \
                            = GameSimulation.gen_next_state(state=state, def_action=def_action, att_action=att_action,
                                                            payoff_matrix=payoff_matrix, adj_matrix=adj_matrix, keep_res=keep_res)
                    next_att_observation = GameSimulation.gen_next_observation(observation=attacker_observation,
                                                                               def_action=def_action,
                                                                               att_action=att_action)
                    def_utility_average += def_immediate_utility
                    att_utility_average += att_immediate_utility

                    state = next_state
                    attacker_observation = next_att_observation
                    num_att -= sum(att_action).item()

        def_utility_average /= n_sample
        att_utility_average /= n_sample

        return def_utility_average.item(), att_utility_average.item()


def get_util_probs(payoff: Iterable)-> list:
    pay = [(r + p) / 2 for r, p in payoff]
    quanta = [math.exp(x) for x in pay]
    qr = [q/sum(quanta) for q in quanta]
    return qr

def _get_util_probs(payoff, version='def'):
    pay = [(r.item() + p.item()) / 2 for r, p in payoff]
    quanta = [math.exp(x) for x in pay]
    qr = [q/sum(quanta) for q in quanta]
    return qr


class GameGeneration(object):
    def __init__(
        self, 
        num_target: int=10, 
        num_res: int=5, 
        groups: int=3,
        graph_type: str='random_scale_free', 
    ):
        self.num_target = num_target
        self.graph_type = graph_type
        self.num_res = num_res
        self.min_value = 1
        self.groups = groups

    # Game Generation
    def gen_game(self):
        # random.seed(seed)
        if self.graph_type == 'random_scale_free':
            # graph = nx.scale_free_graph(self.num_target, seed=seed)
            graph = nx.scale_free_graph(self.num_target)
            adj_matrix = nx.to_numpy_array(graph)
            for i in range(self.num_target):
                for j in range(i + 1, self.num_target):
                    if adj_matrix[i, j] > 0 or adj_matrix[j, i] > 0:
                        adj_matrix[i, j] = np.random.uniform(-0.2, -0.05)
                        adj_matrix[j, i] = adj_matrix[i, j]
                    else:
                        adj_matrix[i, j] = self.min_value
                        adj_matrix[j, i] = self.min_value
                adj_matrix[i, i] = 0

        adj_hat = adj_matrix + np.eye(self.num_target)
        adj_hat = self.normalize(adj_hat)

        adj_matrix = adj_matrix.astype(np.float32)
        adj_hat = adj_hat.astype(np.float32)

        # Generate payoff matrix of the game
        # torch.manual_seed(seed)
        payoff_matrix = np.random.uniform(low=0, high=1, size=(self.num_target, 4)).astype(np.float32)
        payoff_matrix[:, 0] = payoff_matrix[:, 0] * 0.9 + 0.1
        payoff_matrix[:, 1] = payoff_matrix[:, 1] * 0.9 - 1.0
        payoff_matrix[:, 2] = -payoff_matrix[:, 1].copy()
        payoff_matrix[:, 3] = -payoff_matrix[:, 0].copy()

        return payoff_matrix, adj_matrix, adj_hat, self.gen_def_constraints()

    # Generate defender resource constraints
    def gen_def_constraints(self):
        def_constraints = [[] for _ in range(self.groups)]
        count = self.num_res
        group_limit = round(self.num_res/self.groups)
        for g in def_constraints:
            empty_ct = len([x for x in def_constraints if len(x) < 1])
            group_max = count - empty_ct + 1
            if empty_ct == 1:   add_res = count
            else:               add_res = np.random.randint(1, group_max+1)
            for i in range(add_res):
                res = np.random.randint(0, self.num_res)
                while res in (item for group in def_constraints for item in group):
                    res = np.random.randint(0, self.num_res)
                g.append(res)
                count -= 1
                if i+1 >= group_limit: break
        return def_constraints

    # Normalize adj matrix
    def normalize(self, mx: np.ndarray, eps=1e-8):
        """Row-normalize sparse matrix"""
        rowsum = mx.sum(1, keepdims=True)
        return mx/(rowsum+eps)

def check_adj(state, new_loc, adj):
    """Check if the new location (assignment) is valid by checking the adjacency matrix.
    """
    cur_loc = [[i,n] for i,n in enumerate(state[:,0].tolist()) if n > 0]
    old_loc = [] # sequence of assginment for resources, 
                 # e.g. [0, 4, 5] means rsc 1 is moved from target 0, rsc 2 is moved from target 4, rsc 3 is moved from target 5
    for loc in new_loc:
        for i,res in enumerate(cur_loc):
            if res[1] <= 0: # if no more resource at this location, continue
                continue
            elif adj[res[0], loc] < config.MIN_VALUE: # if edge exist, move from old to new, break
                cur_loc[i][1] -= 1
                old_loc.append(res[0])
                break

    check = [x for x in cur_loc if x[1] > 0] # non-empty current location means invalid move exists
    if len(check) > 0:
        old_loc = []
        for i,t in enumerate(state[:, 0]):
            for _ in range(t):
                old_loc.append(i)
        return False, old_loc
    return True, old_loc


def check_constraints(next_loc, def_constraints, reach, test=None):
    for group in def_constraints:
        for res in group:
            pos = (next_loc[res] == 1).nonzero()[0].item()
            res_group = [x for x in group if x != res]
            for other_res in res_group:
                other_pos = (next_loc[other_res] == 1).nonzero()[0].item()
                # if adj_matrix[pos][other_pos] == config.MIN_VALUE:
                if other_pos not in reach[pos]:
                    if test: print("Move is invalid.")
                    return False
    return True


def gen_init_def_pos(num_targ, num_res, def_constraints, reach):
    loc = torch.zeros((num_res, num_targ), dtype=torch.float, device=device)
    for constraint in def_constraints:
        res_positions = []
        while len(res_positions) < 1:
            for j,res in enumerate(constraint):
                if j > 0:
                    neighbor_list = get_valid_moves(num_targ, res_positions, reach)
                    if len(neighbor_list) < 1: 
                        res_positions = []
                        break
                    pos = random.choice(neighbor_list)
                else:
                    pos = random.randint(0, num_targ-1)
                res_positions.append((res, pos))
        for res,pos in res_positions:
            loc[res][pos] = 1

    return loc


def get_valid_moves(num_targ, res_positions, reach):
    neighbor_list = set([x for x in range(num_targ)])
    for _,k in res_positions:
        neighbor_list = neighbor_list & reach[k]
    if len(neighbor_list) < 1:
        print(res_positions, reach)
    return list(neighbor_list)


def gen_action_dict():
    act_codes = list(itertools.combinations(range(config.NUM_TARGET), config.NUM_RESOURCE))
    action_dict = dict(zip(range(len(act_codes)), act_codes))
    return action_dict


if __name__ == "__main__":
    TEST_SETTING = 'game_gen'

    print("Testing...")

    if TEST_SETTING in ['normalize', 'all']:
        print("="*10, "Test - Normalize", "="*10)
        np.random.seed(0)
        mxs = [np.random.rand(3,5) for _ in range(10)]
        game_gen = GameGeneration()
        i = 0
        
        for mx in mxs:
            try:
                mx_tensor = torch.from_numpy(mx)
                # mx_tensor_norm = game_gen._normalize(mx_tensor) # Del already
                mx_norm = game_gen.normalize(mx)
                if not np.allclose(mx_tensor_norm.numpy(), mx_norm):
                    print("Error:", mx_tensor_norm.numpy(), mx_norm)
            except Exception as e:
                print(e)
                print("Error:", mx)
                i += 1
                continue
        print(f"\nPassed {i} tests.")
        print("="*10,"Test ends.","="*10)

    if TEST_SETTING in ['all', 'game_gen']:
        print("="*10, "Test - Generate Game Parameters", "="*10)
        game_gen = GameGeneration(num_target=5, num_res=4)
        params = game_gen.gen_game()
        # params_prime = game_gen._gen_game() # Del already

        for p, p_prime in zip(params, params_prime):
            print(f'{p}, \n{p_prime}')
        print("="*10,"Test ends.","="*10)
