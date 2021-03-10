# -*- coding: utf-8 -*-

import os
import abc
import time
import numpy as np
import pandas as pd
import seaborn as sns
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from gym.envs.toy_text import discrete, frozen_lake, blackjack
import gym
from tqdm import tqdm
import random
from matplotlib.lines import Line2D
import multiprocessing

import gym
import random

from gym import spaces

class GamblerEnv(discrete.DiscreteEnv):

    """Gambler's Problem environment.

    References
    ----------
    .. [1] Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An
       introduction. MIT press, 2018.

    """

    def __init__(self, max_capital=100, p_heads=0.4):

        assert max_capital > 1
        assert 0 <= p_heads <= 1

        p_tails = 1.0 - p_heads

        nS = max_capital + 1
        nA = max_capital // 2

        isd = np.zeros(nS)
        isd[1:-1] = 1.0 / (nS - 2)

        P = {s: {a: [] for a in range(nA)} for s in range(nS)}
        for s in range(nS):
            capital = s
            for a in range(nA):
                stake = a + 1

                if capital == 0:
                    # out of cash
                    P[s][a].append((1.0, 0, 0.0, True))
                if capital == max_capital:
                    # reached goal
                    P[s][a].append((1.0, max_capital, 0.0, True))
                elif stake > min(capital, max_capital - capital):
                    # invalid stake
                    P[s][a].append((1.0, s, 0.0, False))
                else:
                    # coin lands on heads
                    capital_heads = capital + stake
                    if capital_heads < max_capital:
                        P[s][a].append((p_heads, capital_heads, 0.0, False))
                    else:
                        P[s][a].append((p_heads, capital_heads, 1.0, True))

                    # coin lands on tails
                    capital_tails = capital - stake
                    if capital_tails > 0:
                        P[s][a].append((p_tails, capital_tails, 0.0, False))
                    else:
                        P[s][a].append((p_tails, capital_tails, 0.0, True))

        super().__init__(nS, nA, P, isd)

        self.max_capital = max_capital
        self.p_heads = p_heads
        self.p_tails = p_tails

    @property
    def capital(self):
        return self.s

    def __repr__(self):
        return '\n'.join([
            f'Current capital: {self.capital}',
            f'Last stake: {self.lastaction}',
            f'Number of bets: {self.num_steps}'
        ])

    def render(self):
        return print(repr(self))
class DecayVariable:
    def __init__(self, maxV, minV, decay):
        self.maxV = maxV
        self.minV = minV
        self.decay = decay

    def __call__(self, step):
        k = self.minV + (self.maxV - self.minV) * np.exp(-self.decay * step)
        return k

def one_step_lookahead(env, state, V, gamma):
    """Compute action values using a one-step lookahead search."""
    action_values = np.zeros(env.nA)
    for action in range(env.nA):
        for prob, next_state, reward, _ in env.P[state][action]:
            action_values[action] += prob * (reward + gamma * V[next_state])
    return action_values
def argmax(a, atol=1e-16):
    return np.where(np.isclose(a, a.max(), atol=atol))[0][0]

def run_mdp(env, policy, update_value=lambda s, _: s + 1, steps_lim=None, progress_bar=True):
    results = {
        'wins': 0,
        'losses': 0,
    }
    num_episodes = 1000
    cum_rewards = np.zeros(shape=(num_episodes,))
    eps = tqdm(range(num_episodes)) if progress_bar else range(num_episodes)
    for i in eps:
        state = env.reset()
        random.seed(i)
        np.random.seed(i)
        env.seed(i)
        steps = 0
        cum_reward = 0
        while True:
            state, reward, done, _ = env.step(policy[state])
            cum_reward = update_value(cum_reward, reward)
            if done:
                if reward > 0:
                    results['wins'] += 1
                else:
                    results['losses'] += 1
                cum_rewards[i] = cum_reward
                break
            if steps_lim and steps >= steps_lim:
                results['losses'] += 1
                cum_rewards[i] = cum_reward
                break
            steps += 1
    return (
        1.0 * results['wins'] / (results['wins'] + results['losses']),
        cum_rewards
    )
def plot_policy(env, policy, **fig_kwargs):
    """Plot policy on Frozen Lake map."""
    grid_shape = (env.nrow, env.ncol)

    fig = plt.figure(**fig_kwargs)
    fig.clf()
    ax = fig.gca()

    labels = {
        0: r'$\leftarrow$ ',
        1: r'$\downarrow$ ',
        2: r'$\rightarrow$ ',
        3: r'$\uparrow$ '
    }

    annot = np.array([labels[action] for action in policy])
    annot.shape = grid_shape

    cmap = ListedColormap(['seagreen', 'red', 'lightblue', 'black'])

    data = np.zeros(grid_shape, dtype=np.int)
    data[env.desc == b'S'] = 0
    data[env.desc == b'G'] = 1
    data[env.desc == b'F'] = 2
    data[env.desc == b'H'] = 3

    for i in range(env.nrow):
        for j in range(env.ncol):
            k = env.desc[i][j]
            if k == b'G' or k == b'H':
                annot[i][j] = ''


    annot_kws = {
        'fontsize': 25,
        'ha': 'center',
        'va': 'center'
    }

    sns.heatmap(data, annot=annot, fmt='', square=True, cmap=cmap, cbar=False,
                annot_kws=annot_kws, ax=ax)

    return fig
def plotGrid(env, V, policy, num_iters, substr):
    fig = plt.figure()
    ax = fig.gca()
    sns.heatmap(V.reshape((env.nrow, env.ncol)), annot=True, fmt='.2f',
                square=True, cbar=False, vmin=0, vmax=1,
                annot_kws={'fontsize': 13}, ax=ax)
    ax.set_title('Weights Plot'.format(num_iters))
    fig.tight_layout()
    fig.savefig(os.path.join('output', 'FL-HM_{}.png'.format(substr)), bbox_inches='tight')

    fig = plot_policy(env, policy)
    ax = fig.gca()
    ax.set_title('Policy Plot'.format(num_iters))
    fig.tight_layout()
    fig.savefig(os.path.join('output', 'FL-MP_{}.png'.format(substr, num_iters)), bbox_inches='tight')
          
def PolicyIterationTest(env, gamma, theta, max_iters, substr):
    """Find an optimal policy using policy iteration."""
    
    V = np.zeros(env.nS)
    policy = np.random.randint(low=0, high=env.nA, size=env.nS)
    policy_prev = np.copy(policy)
    time_list = []
    errors = []
    mean_list = []
    rewards_list = []
    delta_list = []
    convergence = 0
    converged = False
    iterator = 1
    
    for n_iters in range(1, max_iters + 1):#tqdm(range(1, max_iters + 1)):
        start = time.time()
        policy_value = np.zeros(env.nS)
        delta = 0.0
        for state, action in enumerate(policy):
            for probablity, next_state, reward, info in env.P[state][action]:
                policy_value[state] += probablity * (reward + (gamma * V[next_state]))
            delta = max(delta, abs(V[state] - policy_value[state]))
        V = policy_value
        for state in range(env.nS):
            action_values = one_step_lookahead(env, state, V, gamma)
            policy[state] = np.argmax(action_values)
        delta_list.append(delta)
        time_list.append(time.time() - start)
        error = np.not_equal(policy, policy_prev).sum() / env.nS
        errors.append(error)
        mean_list.append(V.mean())

        if error == 0:
            convergence += 1
        else:
            convergence = 0
        if convergence >= 10:
            converged = True
            break
        _, rewards = run_mdp(env, policy, steps_lim=500, update_value=lambda a,b: a+b, progress_bar=False)
        rewards_list.append(rewards.mean())
        policy_prev = np.copy(policy)
        
        if delta < theta and not converged:
            converged = True
            break
        iterator += 1
    
    string = r"PI Convergence {}  gamma({}), theta({}) reached at {}/{} iterations in {}(s). delta={}, mean={}, reward={}. rewards/iterations={}".format(str(converged), gamma, theta, iterator, max_iters, round(sum(time_list),2), round(min(delta_list),10), round(max(mean_list),4), round(max(rewards_list),4), round(sum(rewards_list)/iterator,3))
    print (string)
    if not os.path.exists('output'):
        os.makedirs('output')
    with open('output/{}'.format(substr) +".txt", 'a+') as f:
        f.write(string + "\n")

    info = {
        'V': V,
        'policy': policy,
        'num_iters': iterator,
        'delta_list': errors,
        'time_list': time_list,
        'mean_list': mean_list,
        'reward_list': rewards_list
    }

    return info
def ValueIterationTest(env, gamma, theta, max_iters, substr):
    """Find an optimal policy using value iteration."""
    V = np.zeros(env.nS)
    policy = np.zeros(env.nS, dtype=np.int)
    policy_prev = np.copy(policy)
    delta_list = []
    time_list = []
    mean_list = []
    rewards_list = []
    erros_list = []
    
    n_iters = 0
    converged = False
    iterator = 0
    for num_iters in range(1, max_iters + 1):#tqdm(range(1, max_iters + 1)):
        start = time.time()
        delta = 0.0
        for state in range(env.nS):
            action_values = one_step_lookahead(env, state, V, gamma)
            action_value = action_values.max()
            delta = max(delta, abs(V[state] - action_value))
            V[state] = action_value
            policy[state] = np.argmax(action_values)
        num_iters, rewards = run_mdp(env, policy, steps_lim=500, update_value=lambda a,b: a+b, progress_bar=False)
        rewards_list.append(rewards.mean())
        mean_list.append(V.mean())
        delta_list.append(delta)
        time_list.append(time.time()-start)

        error = np.not_equal(policy, policy_prev).sum() / env.nS
        erros_list.append(error)
        policy_prev = np.copy(policy)

        if delta < theta and not converged:
            converged = True
            break
        iterator += 1

    string = r"VI Convergence {}  gamma({}), theta({}) reached at {}/{} iterations in {}(s). delta={}, mean={}, reward={}. rewards/iterations={}".format(str(converged), gamma, theta, iterator, max_iters, round(sum(time_list),2), round(min(delta_list),10), round(max(mean_list),4), round(max(rewards_list),4), round(sum(rewards_list)/iterator, 3))
    print (string)
    if not os.path.exists('output'):
        os.makedirs('output')
    with open('output/{}'.format(substr) +".txt", 'a+') as f:
        f.write(string + "\n")

    info = {
        'V': V,
        'policy': policy,
        'num_iters': iterator,
        'delta_list': erros_list,
        'time_list': time_list,
        'mean_list': mean_list,
        'reward_list': rewards_list
    }

    return info
def QLearningTest(env, num_episodes, aMax, aMin, aDecay, eMax, eMin, eDecay, gMax, gMin, gDecay, theta, explore, T, beta, avg_window, substr):
    # https://medium.com/emergent-future/simple-reinforcement-learning-with-tensorflow-part-0-q-learning-with-tables-and-neural-networks-d195264329d0
    # Initialize table with all zeros
    
    def exponential_decay(step, max_value, min_value, decay):
        return min_value + (max_value - min_value) * np.exp(-decay * step)
    
    def bolzmann_action(state):
        p = np.array([Q[(state,x)]/T for x in range(env.action_space.n)])
        prob_actions = np.exp(p) / np.sum(np.exp(p))
        cumulative_probability = 0.0
        choice = random.uniform(0,1)
        for a,pr in enumerate(prob_actions):
            cumulative_probability += pr
            if cumulative_probability > choice:
                return a

    def ucb_action(state):
        curr_q = Q[state, :] + beta * np.sqrt(np.log(N[state, :].sum()) / (2 * N[state, :]))
        best_action = curr_q.argmax()
        N[state, best_action] += 1
        return best_action

    Q = np.zeros([env.observation_space.n,env.action_space.n])
    policy = np.zeros(shape=(env.nS,), dtype=int)
    V = np.empty(env.nS)
    N = np.ones(shape=(env.nS, env.nA), dtype=int)
    
    # create lists to contain total rewards and steps per episode
    time_list = []
    times_list = []
    error_list = []
    errors_list = []
    reward_list = []
    rewards_list = []
    delta_list = []
    deltas_list = []
    i = 0
    converged = False
    for episode in range(1, num_episodes + 1):#tqdm(range(1, num_episodes+1)):
        # Reset environment and get first new observation
        state = env.reset()
        rewardAll = 0
        j = 0
        delta = 0
        newQ = Q.copy()
        start = time.time()
        alpha = exponential_decay(episode, aMax, aMin, aDecay)
        gamma = exponential_decay(episode, gMax, gMin, gDecay)
        epsilon = exponential_decay(episode, eMax, eMin, eDecay)
        # The Q-Table learning algorithm
        
        while True:
            j += 1
            # Choose an action by greedily (with noise) picking from Q table
            # action = np.argmax(Q[state,:] + np.random.randn(1, env.action_space.n)*(1./(episode+1)))
            
            rng = np.random.RandomState()

            if(explore is "boltzman"):
                action = bolzmann_action(state)
            elif (explore is "ucb"):
                action = ucb_action(state)
            elif(explore is "epsilon"):
                action = rng.randint(Q[state].size) if rng.rand() < epsilon else Q[state].argmax()
            elif(explore is "random"):
                action = rng.randint(Q[state].size)
            elif(explore is "greedy"):
                action = Q[state].argmax()
            
            # Get new state and reward from environment
            new_state, reward, done, _ = env.step(action)
            # Update Q-Table with new knowledge
            pQ = Q[state, action]
            newQ[state,action] = pQ + alpha*(reward + gamma*np.max(Q[new_state, :]) - pQ)
            delta = max(delta, np.abs(newQ[state, action] - pQ))
            rewardAll += reward
            state = new_state
            if done == True:
                break
            
        delta_list.append(delta)
        error_list.append(np.not_equal(newQ.argmax(axis=1), policy).sum()/env.nS)
        reward_list.append(rewardAll)
        time_list.append(time.time()-start)
        if  0 < sum(delta_list)/episode < theta:
            converged = True
            break
        if episode % avg_window == 0:
            temp = delta_list[-avg_window:]
            deltas_list.append(sum(temp)/avg_window)
            temp = error_list[-avg_window:]
            errors_list.append(sum(temp)/avg_window)
            temp = reward_list[-avg_window:]
            rewards_list.append(sum(temp)/avg_window)
            temp = time_list[-avg_window:]
            times_list.append(sum(temp)/avg_window)
        Q = newQ.copy()
        policy = Q.argmax(axis=1)
        i+=1
    
    for state in range(env.nS):
        action_value = Q[state, Q[state].argmax()]
        V[state] = action_value

    if(explore is "boltzman"):
        string = r"Q {} T({}) Convergence {} in {} episodes. avg.error={}, avg.delta={}, avg.reward={}".format(explore, T, str(converged), i, round(sum(error_list)/i, 3), round(sum(delta_list)/i, 3), round(sum(reward_list)/i, 3))
    elif(explore is "ucb"):
        string = r"Q {} beta({}) Convergence {} in {} episodes. avg.error={}, avg.delta={}, avg.reward={}".format(explore, beta, str(converged), i, round(sum(error_list)/i, 3), round(sum(delta_list)/i, 3), round(sum(reward_list)/i, 3))
    elif(explore is "epsilon"):
        string = r"Q {} eMax({}) eMin({}) eDecay({}) Convergence {} in {} episodes. avg.error={}, avg.delta={}, avg.reward={}".format(explore, eMax, eMin, eDecay, str(converged), i, round(sum(error_list)/i, 3), round(sum(delta_list)/i, 3), round(sum(reward_list)/i, 3))
    elif(explore is "random" or explore is "greedy"):
        string = r"Q {} aMax({}) aMin({}) aDecay({}) gMax({}) gMin({}) gDecay({}) Convergence {} in {} episodes. avg.error={}, avg.delta={}, avg.reward={}\n".format(explore, aMax, aMin, aDecay, gMax, gMin, gDecay, str(converged), i, round(sum(error_list)/i, 3), round(sum(delta_list)/i, 3), round(sum(reward_list)/i, 3))

    print (string)
    if not os.path.exists('output'):
        os.makedirs('output')
    with open('output/{}'.format(substr) +".txt", 'a+') as f:
        f.write(string + "\n")

    return (
        {'policy': policy,
        'V': V, 
        'time_list': time_list, 
        'times_list': times_list, 
        'error_list': error_list,
        'errors_list': errors_list,
        'delta_list': delta_list, 
        'deltas_list': deltas_list,
        'reward_list': reward_list, 
        'rewards_list': rewards_list,
        'num_episodes': i}
    ) 

def FrozenLakeVI(substr, env, max_itr):
    """Run value iteration on 4x4 Frozen Lake environment."""
    
    gamma_arr = [1.0, 0.7, 0.4]
    theta_arr = [1e-7, 1e-5, 1e-3]

    color = ['k', 'b', 'g']

    best = 0
    bestInfo = {}

    fig = plt.figure(figsize=(8,12))
    plots = []
    subplot1 = []
    subplot1.append(fig.add_axes([0.1, 0.14, 0.375, 0.12]))
    subplot1.append(fig.add_axes([0.1, 0.26, 0.375, 0.12], xticklabels=[]))
    subplot1.append(fig.add_axes([0.1, 0.38, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot1)
    subplot2 = []
    subplot2.append(fig.add_axes([0.1, 0.62, 0.375, 0.12]))
    subplot2.append(fig.add_axes([0.1, 0.74, 0.375, 0.12], xticklabels=[]))
    subplot2.append(fig.add_axes([0.1, 0.86, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot2)
    subplot3 = []
    subplot3.append(fig.add_axes([0.6, 0.14, 0.375, 0.12]))
    subplot3.append(fig.add_axes([0.6, 0.26, 0.375, 0.12], xticklabels=[]))
    subplot3.append(fig.add_axes([0.6, 0.38, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot3)
    subplot4 = []
    subplot4.append(fig.add_axes([0.6, 0.62, 0.375, 0.12]))
    subplot4.append(fig.add_axes([0.6, 0.74, 0.375, 0.12], xticklabels=[]))
    subplot4.append(fig.add_axes([0.6, 0.86, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot4)

    for i in range(len(gamma_arr)):
        for j in range(len(theta_arr)):
            env.reset()
            info = ValueIterationTest(env, gamma_arr[i], theta_arr[j], max_itr, substr)
            plots[0][i].plot(range(len(info['delta_list'])), info['delta_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[1][i].plot(range(len(info['mean_list'])), info['mean_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[2][i].plot(range(len(info['time_list'])), info['time_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[3][i].plot(range(len(info['reward_list'])), info['reward_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[0][i].plot(len(info['delta_list'])-1,info['delta_list'][-1], 'rD', lw=5)
            plots[1][i].plot(len(info['mean_list'])-1,info['mean_list'][-1], 'rD', lw=5)
            plots[2][i].plot(len(info['time_list'])-1,info['time_list'][-1], 'rD', lw=5)
            plots[3][i].plot(len(info['reward_list'])-1,info['reward_list'][-1], 'rD', lw=5)
            temp = sum(info['reward_list'])/info['num_iters']
            if temp > best:
                best = temp
                bestInfo = info
            if not bestInfo:
                bestInfo = info

    plots[0][0].set_xlabel('Iteration')
    plots[0][1].set_ylabel('Error')
    plots[0][2].set_title('Error vs Iteration')
    plots[1][0].set_xlabel('Iteration')
    plots[1][1].set_ylabel('Time (s)')
    plots[1][2].set_title('Runtime vs Iteration')
    plots[2][0].set_xlabel('Iteration')
    plots[2][1].set_ylabel('Average Value')
    plots[2][2].set_title('Avg Value vs Iteration')
    plots[3][0].set_xlabel('Iteration')
    plots[3][1].set_ylabel('Average Reward')
    plots[3][2].set_title('Avg Reward vs Iteration')

    plots[0][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[0][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[0][2].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[1][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[1][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[1][2].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[2][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[2][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[2][2].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[3][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[3][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[3][2].legend(handlelength=0, handletextpad=0, loc="upper right")

    custom_lines = [Line2D([0], [0], color=color[0]),
                    Line2D([0], [0], color=color[1]),
                    Line2D([0], [0], color=color[2])]
                    
    fig.legend(custom_lines, [r'$\theta={}$'.format(theta_arr[0]), 
                            r'$\theta={}$'.format(theta_arr[1]),  
                            r'$\theta={}$'.format(theta_arr[2])], loc="center right")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join('output', 'FL-{}.png'.format(substr)), bbox_inches='tight')
    plt.close(fig)
    if not isinstance(env, GamblerEnv):
        plotGrid(env, bestInfo['V'], bestInfo['policy'], bestInfo['num_iters'], substr)
def FrozenLakePI(substr, env, max_itr):
    """Run policy iteration on 4x4 Frozen Lake environment."""
    
    gamma_arr = [1.0, 0.7, 0.4]
    theta_arr = [1e-7, 1e-5, 1e-3]
    
    color = ['k', 'b', 'g']

    best = 0
    bestInfo = {}
    fig = plt.figure(figsize=(8,12))
    plots = []
    subplot1 = []
    subplot1.append(fig.add_axes([0.1, 0.14, 0.375, 0.12]))
    subplot1.append(fig.add_axes([0.1, 0.26, 0.375, 0.12], xticklabels=[]))
    subplot1.append(fig.add_axes([0.1, 0.38, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot1)
    subplot2 = []
    subplot2.append(fig.add_axes([0.1, 0.62, 0.375, 0.12]))
    subplot2.append(fig.add_axes([0.1, 0.74, 0.375, 0.12], xticklabels=[]))
    subplot2.append(fig.add_axes([0.1, 0.86, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot2)
    subplot3 = []
    subplot3.append(fig.add_axes([0.6, 0.14, 0.375, 0.12]))
    subplot3.append(fig.add_axes([0.6, 0.26, 0.375, 0.12], xticklabels=[]))
    subplot3.append(fig.add_axes([0.6, 0.38, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot3)
    subplot4 = []
    subplot4.append(fig.add_axes([0.6, 0.62, 0.375, 0.12]))
    subplot4.append(fig.add_axes([0.6, 0.74, 0.375, 0.12], xticklabels=[]))
    subplot4.append(fig.add_axes([0.6, 0.86, 0.375, 0.12], xticklabels=[]))
    plots.append(subplot4)

    for i in range(len(gamma_arr)):
        for j in range(len(theta_arr)):
            env.reset()
            info = PolicyIterationTest(env, gamma_arr[i], theta_arr[j], max_itr, substr)
            plots[0][i].plot(range(len(info['delta_list'])), info['delta_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[1][i].plot(range(len(info['mean_list'])), info['mean_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[2][i].plot(range(len(info['time_list'])), info['time_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[3][i].plot(range(len(info['reward_list'])), info['reward_list'], color=color[j], label=r'$\gamma={}$'.format(gamma_arr[i]) if j is 0 else '')
            plots[0][i].plot(len(info['delta_list'])-1,info['delta_list'][-1], 'rD', lw=5)
            plots[1][i].plot(len(info['mean_list'])-1,info['mean_list'][-1], 'rD', lw=5)
            plots[2][i].plot(len(info['time_list'])-1,info['time_list'][-1], 'rD', lw=5)
            plots[3][i].plot(len(info['reward_list'])-1,info['reward_list'][-1], 'rD', lw=5)
            temp = sum(info['reward_list'])/info['num_iters']
            if temp > best:
                best = temp
                bestInfo = info
            if not bestInfo:
                bestInfo = info
    
    plots[0][0].set_xlabel('Iteration')
    plots[0][1].set_ylabel('Error')
    plots[0][2].set_title('Error vs Iteration')
    plots[1][0].set_xlabel('Iteration')
    plots[1][1].set_ylabel('Time (s)')
    plots[1][2].set_title('Runtime vs Iteration')
    plots[2][0].set_xlabel('Iteration')
    plots[2][1].set_ylabel('Average Value')
    plots[2][2].set_title('Avg Value vs Iteration')
    plots[3][0].set_xlabel('Iteration')
    plots[3][1].set_ylabel('Average Reward')
    plots[3][2].set_title('Avg Reward vs Iteration')
    plots[0][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[0][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[0][2].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[1][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[1][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[1][2].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[2][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[2][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[2][2].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[3][0].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[3][1].legend(handlelength=0, handletextpad=0, loc="upper right")
    plots[3][2].legend(handlelength=0, handletextpad=0, loc="upper right")

    custom_lines = [Line2D([0], [0], color=color[0]),
                    Line2D([0], [0], color=color[1]),
                    Line2D([0], [0], color=color[2])]
                    
    fig.legend(custom_lines, [r'$\theta={}$'.format(theta_arr[0]), 
                            r'$\theta={}$'.format(theta_arr[1]),  
                            r'$\theta={}$'.format(theta_arr[2])], loc="center")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join('output', 'FL-{}.png'.format(substr)), bbox_inches='tight')
    plt.close(fig)

    if not isinstance(env, GamblerEnv):
        plotGrid(env, bestInfo['V'], bestInfo['policy'], bestInfo['num_iters'], substr)
def FrozenLakeQ(substr, env, num_episodes, avg_window): 
    """Run Q-learning on 4x4 Frozen Lake environment."""

    # Learning Rate
    aMax = 0.3
    aMin = 0.01
    aDecay = 0.0001

    # Ratio of explore vs exploit
    eMax = [1.0, 0.9, 0.8]
    eMin = [0.5, 0.4, 0.3]
    eDecay = [0.001, 0.0001, 0.00001]

    # Discount Factor (1 = future, 0 = immediate)
    gMax = 1.0
    gMin = 0.8
    gDecay = 0.0001

    theta = 0.00001
    exploration = ["ucb", "boltzman", "epsilon"]
    base_action = ["random", "greedy"]
    tests = ["LearningRate", "DiscountFactor"]

    T = [1, 0.1, 0.01]
    Beta = [1, 0.7, 0.5]

    color = ['k', 'b', 'g']

    # Explore Random Action Learning Rate and Discount Factor
    # Learning Rate
    lrMax = [0.3, 0.4, 0.5]
    lrMin = [0.01, 0.1, 0.2]
    lrDecay = [0.001, 0.0001, 0.00001]
    dfMax = [1.0, 0.9, 0.8]
    dfMin = [0.8, 0.7, 0.6]
    dfDecay = [0.001, 0.0001, 0.00001]

    if not os.path.exists('output'):
        os.makedirs('output')

    best = 0
    bestInfo = {}
    for explore in base_action:
        for test in tests:
            
            fig = plt.figure(figsize=(8,12))
            plots = []
            subplot1 = []
            subplot1.append(fig.add_axes([0.1, 0.14, 0.375, 0.12]))
            subplot1.append(fig.add_axes([0.1, 0.26, 0.375, 0.12], xticklabels=[]))
            subplot1.append(fig.add_axes([0.1, 0.38, 0.375, 0.12], xticklabels=[]))
            plots.append(subplot1)
            subplot2 = []
            subplot2.append(fig.add_axes([0.1, 0.62, 0.375, 0.12]))
            subplot2.append(fig.add_axes([0.1, 0.74, 0.375, 0.12], xticklabels=[]))
            subplot2.append(fig.add_axes([0.1, 0.86, 0.375, 0.12], xticklabels=[]))
            plots.append(subplot2)
            subplot3 = []
            subplot3.append(fig.add_axes([0.6, 0.14, 0.375, 0.12]))
            subplot3.append(fig.add_axes([0.6, 0.26, 0.375, 0.12], xticklabels=[]))
            subplot3.append(fig.add_axes([0.6, 0.38, 0.375, 0.12], xticklabels=[]))
            plots.append(subplot3)
            subplot4 = []
            subplot4.append(fig.add_axes([0.6, 0.62, 0.375, 0.12]))
            subplot4.append(fig.add_axes([0.6, 0.74, 0.375, 0.12], xticklabels=[]))
            subplot4.append(fig.add_axes([0.6, 0.86, 0.375, 0.12], xticklabels=[]))
            plots.append(subplot4)

            string = substr + " " + explore + " " + test + "Test"
            print (string)
            with open('output/{}'.format(substr) +".txt", 'a+') as f:
                f.write(string + "\n")
            for i in range(3):
                for j in range(3):
                    env.reset()
                    label = ''
                    if test is "LearningRate":
                        info = QLearningTest(env, num_episodes, lrMax[i], lrMin[i], lrDecay[j], eMax[i], eMin[i], eDecay[i], gMax, gMin, gDecay, theta, explore, T[i], Beta[i], avg_window, substr)
                        label = r'$\alpha\nabla={}$'.format(lrDecay[i]) if j is 0 else ''
                    else:
                        info = QLearningTest(env, num_episodes, aMax, aMin, aDecay, eMax[i], eMin[i], eDecay[i], dfMax[i], dfMin[i], dfDecay[j], theta, explore, T[i], Beta[i], avg_window, substr)
                        label = r'$\gamma\nabla={}$'.format(dfDecay[i])  if j is 0 else ''
                    temp = sum(info['reward_list'])/info['num_episodes']
                    if temp > best:
                        best = temp
                        bestInfo = info
                    if not bestInfo:
                        bestInfo = info
                    
                    plots[0][i].plot(range(len(info['deltas_list'])), info['deltas_list'], color=color[j], label = label)
                    plots[1][i].plot(range(len(info['rewards_list'])), info['rewards_list'], color=color[j], label = label)
                    plots[2][i].plot(range(len(info['times_list'])), info['times_list'], color=color[j], label = label)
                    plots[3][i].plot(range(len(info['errors_list'])), info['errors_list'], color=color[j], label = label)
                    plots[0][i].plot(len(info['deltas_list'])-1,info['deltas_list'][-1], 'rD', lw=5)
                    plots[1][i].plot(len(info['rewards_list'])-1,info['rewards_list'][-1], 'rD', lw=5)
                    plots[2][i].plot(len(info['times_list'])-1,info['times_list'][-1], 'rD', lw=5)
                    plots[3][i].plot(len(info['errors_list'])-1,info['errors_list'][-1], 'rD', lw=5)
            
            plots[0][0].set_xlabel('Episodes')
            plots[0][1].set_ylabel('Delta')
            plots[0][2].set_title('Delta vs Episodes')
            plots[1][0].set_xlabel('Episodes')
            plots[1][1].set_ylabel('Time (s)')
            plots[1][2].set_title('Runtime vs Iteration')
            plots[2][0].set_xlabel('Episodes')
            plots[2][1].set_ylabel('Rewards')
            plots[2][2].set_title('Rewards vs Episodes')
            plots[3][0].set_xlabel('Episodes')
            plots[3][1].set_ylabel('Error')
            plots[3][2].set_title('Error vs Episodes')

            plots[0][0].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[0][1].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[0][2].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[1][0].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[1][1].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[1][2].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[2][0].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[2][1].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[2][2].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[3][0].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[3][1].legend(handlelength=0, handletextpad=0, loc="upper right")
            plots[3][2].legend(handlelength=0, handletextpad=0, loc="upper right")

            custom_lines = [Line2D([0], [0], color=color[0]),
                            Line2D([0], [0], color=color[1]),
                            Line2D([0], [0], color=color[2])]
            
            if test is "LearningRate":
                fig.legend(custom_lines, [r'$\alpha\uparrow={}$'.format(lrMax[0]) + ' ' + r'$\alpha\downarrow={}$'.format(lrMin[0]), 
                                        r'$\alpha\uparrow={}$'.format(lrMax[1]) + ' ' + r'$\alpha\downarrow={}$'.format(lrMin[1]),  
                                        r'$\alpha\uparrow={}$'.format(lrMax[2]) + ' ' + r'$\alpha\downarrow={}$'.format(lrMin[2])], loc="center")
            else:
                fig.legend(custom_lines, [r'$\gamma\uparrow={}$'.format(dfMax[0]) + ' ' + r'$\alpha\downarrow={}$'.format(dfMin[0]),
                                        r'$\gamma\uparrow={}$'.format(dfMax[1]) + ' ' + r'$\alpha\downarrow={}$'.format(dfMin[1]),  
                                        r'$\gamma\uparrow={}$'.format(dfMax[2]) + ' ' + r'$\alpha\downarrow={}$'.format(dfMin[2])], loc="center")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(os.path.join('output', 'FL-{}_{}_{}.png'.format(str(explore),str(test), substr)), bbox_inches='tight')
            plt.clf()
            plt.close(fig)

    plotGrid(env, bestInfo['V'], bestInfo['policy'], bestInfo['num_episodes'], substr+'base')

    fig, ax = plt.subplots(2, 2, figsize=(15, 15))
    best = 0
    bestInfo = {}
    for explore in exploration:
        string = substr + " " + explore + " Test"
        print (string)
        with open('output/{}'.format(substr) +".txt", 'a+') as f:
            f.write(string + "\n")

        fig = plt.figure(figsize=(8,12))
        plots = []
        subplot1 = []
        subplot1.append(fig.add_axes([0.1, 0.14, 0.375, 0.12]))
        subplot1.append(fig.add_axes([0.1, 0.26, 0.375, 0.12], xticklabels=[]))
        subplot1.append(fig.add_axes([0.1, 0.38, 0.375, 0.12], xticklabels=[]))
        plots.append(subplot1)
        subplot2 = []
        subplot2.append(fig.add_axes([0.1, 0.62, 0.375, 0.12]))
        subplot2.append(fig.add_axes([0.1, 0.74, 0.375, 0.12], xticklabels=[]))
        subplot2.append(fig.add_axes([0.1, 0.86, 0.375, 0.12], xticklabels=[]))
        plots.append(subplot2)
        subplot3 = []
        subplot3.append(fig.add_axes([0.6, 0.14, 0.375, 0.12]))
        subplot3.append(fig.add_axes([0.6, 0.26, 0.375, 0.12], xticklabels=[]))
        subplot3.append(fig.add_axes([0.6, 0.38, 0.375, 0.12], xticklabels=[]))
        plots.append(subplot3)
        subplot4 = []
        subplot4.append(fig.add_axes([0.6, 0.62, 0.375, 0.12]))
        subplot4.append(fig.add_axes([0.6, 0.74, 0.375, 0.12], xticklabels=[]))
        subplot4.append(fig.add_axes([0.6, 0.86, 0.375, 0.12], xticklabels=[]))
        plots.append(subplot4)

        for i in range(3 if explore is "epsilon" else 1):
            for j in range(3):
                env.reset()

                info = QLearningTest(env, num_episodes, aMax, aMin, aDecay, eMax[j], eMin[j], eDecay[i], gMax, gMin, gDecay, theta, explore, T[j], Beta[j], avg_window, substr)
                
                temp = sum(info['reward_list'])/info['num_episodes']
                if temp > best:
                    best = temp
                    bestInfo = info
                if not bestInfo:
                    bestInfo = info
                
                label = ''            
                if explore is "epsilon":
                    label = r'$\epsilon\nabla={}$'.format(lrDecay[i]) if j is 0 else ''
                    plots[0][i].plot(range(len(info['deltas_list'])), info['deltas_list'], color=color[j], label=label)
                    plots[1][i].plot(range(len(info['rewards_list'])), info['rewards_list'], color=color[j], label=label)
                    plots[2][i].plot(range(len(info['times_list'])), info['times_list'], color=color[j], label=label)
                    plots[3][i].plot(range(len(info['errors_list'])), info['errors_list'], color=color[j], label=label)
                    plots[0][i].plot(len(info['deltas_list'])-1,info['deltas_list'][-1], 'rD', lw=5)
                    plots[1][i].plot(len(info['rewards_list'])-1,info['rewards_list'][-1], 'rD', lw=5)
                    plots[2][i].plot(len(info['times_list'])-1,info['times_list'][-1], 'rD', lw=5)
                    plots[3][i].plot(len(info['errors_list'])-1,info['errors_list'][-1], 'rD', lw=5)
                else:
                    label = 'T={}'.format(T[j]) if explore is "boltzman" else r'$\beta={}'.format(T[j])
                    plots[0][j].plot(range(len(info['deltas_list'])), info['deltas_list'], color=color[j], label=label)
                    plots[1][j].plot(range(len(info['rewards_list'])), info['rewards_list'], color=color[j], label=label)
                    plots[2][j].plot(range(len(info['times_list'])), info['times_list'], color=color[j], label=label)
                    plots[3][j].plot(range(len(info['errors_list'])), info['errors_list'], color=color[j], label=label)

        plots[0][0].set_xlabel('Episodes')
        plots[0][1].set_ylabel('Delta')
        plots[0][2].set_title('Delta vs Episodes')
        plots[1][0].set_xlabel('Episodes')
        plots[1][1].set_ylabel('Time (s)')
        plots[1][2].set_title('Runtime vs Iteration')
        plots[2][0].set_xlabel('Episodes')
        plots[2][1].set_ylabel('Rewards')
        plots[2][2].set_title('Rewards vs Episodes')
        plots[3][0].set_xlabel('Episodes')
        plots[3][1].set_ylabel('Error')
        plots[3][2].set_title('Error vs Episodes')

        plots[0][0].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[0][1].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[0][2].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[1][0].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[1][1].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[1][2].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[2][0].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[2][1].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[2][2].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[3][0].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[3][1].legend(handlelength=0, handletextpad=0, loc="upper right")
        plots[3][2].legend(handlelength=0, handletextpad=0, loc="upper right")          
            
        if explore is "epsilon":  
            custom_lines = [Line2D([0], [0], color=color[0]),
                            Line2D([0], [0], color=color[1]), 
                            Line2D([0], [0], color=color[2])]
                            
            fig.legend(custom_lines, [r'$\epsilon\uparrow={}$'.format(eMax[0]) + ' ' + r'$\epsilon\downarrow={}$'.format(eMin[0]), 
                                    r'$\epsilon\uparrow={}$'.format(eMax[1]) + ' ' + r'$\epsilon\downarrow={}$'.format(eMin[1]),  
                                    r'$\epsilon\uparrow={}$'.format(eMax[2]) + ' ' + r'$\epsilon\downarrow={}$'.format(eMin[2])], loc="center")

        if not os.path.exists('output'):
            os.makedirs('output')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fig.savefig(os.path.join('output', 'FL-{}_{}.png'.format(str(explore), substr)), bbox_inches='tight')
        plt.close(fig)

    plotGrid(env, bestInfo['V'], bestInfo['policy'], bestInfo['num_episodes'], substr+'explore')

def runQ(env, n_repeats, grid, frozen, iterations, window):
    for i in range(n_repeats):
        string = "FrozenLakeQ{}-{}@{}".format(i, grid, frozen)
        print (string)
        substr = "Q{}x{}_{}".format(grid, frozen, i)
        with open('output/{}'.format(substr) +".txt", 'a+') as f:
            f.write(string + "\n")
        FrozenLakeQ(substr, env, iterations, window)
def runP(env, n_repeats, grid, frozen, iterations):
    for i in range(n_repeats):
        string = "FrozenLakeP{}-{}@{}".format(i, grid, frozen)
        print (string)
        substr = "P{}x{}_{}".format(grid, frozen, i)
        with open('output/{}'.format(substr) +".txt", 'a+') as f:
            f.write(string + "\n")
        FrozenLakePI(substr, env, iterations)
def runV(env, n_repeats, grid, frozen, iterations):
    for i in range(n_repeats):
        string = "FrozenLakeV{}-{}@{}".format(i, grid, frozen)
        print (string)
        substr = "V{}x{}_{}".format(grid, frozen, i)
        with open('output/{}'.format(substr) +".txt", 'a+') as f:
            f.write(string + "\n")
        FrozenLakeVI(substr, env, iterations)

def GamblerValueIteration(env, substr, p_heads):
    """Run value iteration for various number of iterations."""
    gamma = 1.0
    theta = 1e-8

    results = {}
    best_Cumulative_rewards = 0
    best_cumulative_rewards_iter = 0
    for max_iters in [1, 2, 3, 5, 8, 13, 21, 34, 55]:
        string = "VI {} Iterations test".format(max_iters)
        print (string)
        with open('output/{}'.format(substr) +".txt", 'a+') as f:
            f.write(string + "\n")
        info = ValueIterationTest(env, gamma, theta, max_iters, substr)
        results[max_iters] = (info['policy'], info['V'])
        if sum(info['reward_list']) > best_Cumulative_rewards:
            best_Cumulative_rewards = sum(info['reward_list'])
            print ("Best={}".format(best_Cumulative_rewards))
            best_cumulative_rewards_iter = max_iters

    fig = plt.figure(figsize=(6.5, 4))
    ax = fig.gca()
    for max_iters, (policy, V) in results.items():
        ax.plot(np.arange(1, 100), V[1:100], label=f'max_iters {max_iters}')
    ax.legend(handlelength=1.7)
    ax.set_xlabel('Capital')
    ax.set_ylabel('Value\nestimates')
    ax.set_xticks([1, 25, 50, 75, 99])
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    fig.savefig(os.path.join('output', '{}{}_vi_values.png'.format(substr, p_heads)))

    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca()
    ax.step(np.arange(1, 100), results[max_iters][0][1:100] + 1, label=f'max_iters {max_iters}')
    ax.set_xlabel('Capital')
    ax.set_ylabel('Final\npolicy\n(stake)')
    ax.set_xticks([1, 25, 50, 75, 99])
    ax.set_yticks([1, 10, 20, 30, 40, 50])
    fig.savefig(os.path.join('output', '{}{}_vi_policy.png'.format(substr, p_heads)))
def GamblerCapitalValue(env, substr, p_heads):

    max_evals = 100
    max_iters = 100

    gamma = 1.0
    theta = 1e-8

    # Learning Rate
    aMax = 0.2
    aMin = 0.0
    aDecay = 0.001

    # Ratio of explore vs exploit
    eMax = 0.5
    eMin = 0.3
    eDecay = 0.001

    # Discount Factor (1 = future, 0 = immediate)
    gMax = 1.0
    gMin = 0.8
    gDecay = 0.0001
    theta = 0.00001

    vi_info = ValueIterationTest(env, gamma, theta, max_iters, substr)
    pi_info = PolicyIterationTest(env, gamma, theta, max_iters, substr)
    ql_info = QLearningTest(env, max_evals, aMax, aMin, aDecay, eMax, eMin, eDecay, gMax, gMin, gDecay, theta, "epsilon", 1, 1, 1, substr)
    
    vi_policy = vi_info['policy']
    pi_policy = pi_info['policy']
    ql_policy = ql_info['policy']

    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca()
    ax.step(np.arange(1, env.max_capital), vi_policy[1:env.max_capital] + 1, color='k')
    ax.set_xlabel('Capital')
    ax.set_ylabel('Final\npolicy\n(stake)', rotation=0, labelpad=30, va='center')
    ax.set_title('Value iteration')
    fig.savefig(os.path.join('output', '{}_vi_policy_{}.png'.format(substr, p_heads)))

    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca()
    ax.step(np.arange(1, env.max_capital), pi_policy[1:env.max_capital] + 1, color='k')
    ax.set_xlabel('Capital')
    ax.set_ylabel('Final\npolicy\n(stake)', rotation=0, labelpad=30, va='center')
    ax.set_title('Policy iteration')
    fig.savefig(os.path.join('output', '{}_pi_policy_{}.png'.format(substr, p_heads)))

    fig = plt.figure(figsize=(6, 3))
    ax = fig.gca()
    ax.step(np.arange(1, env.max_capital), ql_policy[1:env.max_capital] + 1, color='k')
    ax.set_xlabel('Capital')
    ax.set_ylabel('Final\npolicy\n(stake)', rotation=0, labelpad=30, va='center')
    ax.set_title('Q-learning')
    fig.savefig(os.path.join('output', '{}_ql_policy_{}.png'.format(substr, p_heads)))

    cumReward = run_mdp(env, vi_policy)
    string = r'$\Epsilon$'+'VRewards={}'.format(cumReward)
    with open('output/{}'.format(substr) +".txt", 'a+') as f:
        f.write(string + "\n")

    cumReward = run_mdp(env, pi_policy)
    string = r'$\Epsilon$'+'PRewards={}'.format(cumReward)
    with open('output/{}'.format(substr) +".txt", 'a+') as f:
        f.write(string + "\n")

    cumReward = run_mdp(env, ql_policy)
    string = r'$\Epsilon$'+'QRewards={}'.format(cumReward)
    with open('output/{}'.format(substr) +".txt", 'a+') as f:
        f.write(string + "\n")
def GamblerTime(substr):

    # Learning Rate
    aMax = 0.2
    aMin = 0.0
    aDecay = 0.001

    # Ratio of explore vs exploit
    eMax = 0.5
    eMin = 0.3
    eDecay = 0.001

    # Discount Factor (1 = future, 0 = immediate)
    gMax = 1.0
    gMin = 0.8
    gDecay = 0.0001
    theta = 0.00001

    gamma = 1.0
    theta = 1e-8
    max_evals = 1000
    max_iters = 1000
    seed = 0
    num_repeats = 5

    rng = np.random.RandomState(seed=seed)

    sizes = [16, 32, 64, 128, 256, 512, 1024]
    vi_time = np.empty((len(sizes), num_repeats))
    pi_time = np.empty_like(vi_time)
    ql_time = np.empty_like(pi_time)
    for i, size in enumerate(sizes):
        for j in range(num_repeats):
            print(f"Gamber's problem with capital {size}, repeat {j + 1}")
            env = GamblerEnv(max_capital=size, p_heads=0.4)
            
            vi_info = ValueIterationTest(env, gamma, theta, max_iters, substr)
            pi_info = PolicyIterationTest(env, gamma, theta, max_iters, substr)           
            ql_info = QLearningTest(env, max_evals, aMax, aMin, aDecay, eMax, eMin, eDecay, gMax, gMin, gDecay, theta, "epsilon", 1, 1, 1, substr)  
            
            vi_time[i, j] = np.sum(vi_info['time_list'])
            pi_time[i, j] = np.sum(pi_info['time_list'])
            ql_time[i, j] = np.sum(ql_info['time_list'])


    fig = plt.figure()
    ax = fig.gca()
    line, = ax.plot(sizes, vi_time.mean(axis=1), label='Value iteration')
    color = line.get_color()
    ax.fill_between(sizes,
                    vi_time.mean(axis=1) + vi_time.std(axis=1),
                    vi_time.mean(axis=1) - vi_time.std(axis=1),
                    color=color,
                    alpha=0.15)
    line, = ax.plot(sizes, pi_time.mean(axis=1), label='Policy iteration')
    color = line.get_color()
    ax.fill_between(sizes,
                    pi_time.mean(axis=1) + pi_time.std(axis=1),
                    pi_time.mean(axis=1) - pi_time.std(axis=1),
                    color=color,
                    alpha=0.15)
    line, = ax.plot(sizes, ql_time.mean(axis=1), label='Q-Learning')
    color = line.get_color()
    ax.fill_between(sizes,
                    ql_time.mean(axis=1) + ql_time.std(axis=1),
                    ql_time.mean(axis=1) - ql_time.std(axis=1),
                    color=color,
                    alpha=0.15)
    ax.legend()
    ax.set_title("Gambler's Problem")
    ax.set_xlabel('Max capital')
    ax.set_ylabel('Runtime [s]')
    fig.savefig(os.path.join('output', '{}_time.png'.format(substr)))

def main():
    """Run Forzen Lake Value Iteration"""
 
    if not os.path.exists('output'):
        os.makedirs('output')

    pHeads = [0.4, 0.5, 0.6]
    for p in pHeads:
        GamblerValueIteration(GamblerEnv(p_heads=p), "GamblerHProb", p)

    maxCapital = [512, 777, 1023, 1024, 1025]
    for maxCap in maxCapital:
        GamblerCapitalValue(GamblerEnv(max_capital=maxCap), "GamblerCapVal", maxCap)

    GamblerTime("GamblerTime")
    
    env405 = gym.make("FrozenLake-v0", desc=frozen_lake.generate_random_map(size=4, p=0.5))
    env408 = gym.make("FrozenLake-v0", desc=frozen_lake.generate_random_map(size=4, p=0.8))
    env805 = gym.make("FrozenLake-v0", desc=frozen_lake.generate_random_map(size=8, p=0.5))
    env808 = gym.make("FrozenLake-v0", desc=frozen_lake.generate_random_map(size=8, p=0.8))
    runQ405 = multiprocessing.Process(target=runQ, args=(env405, 5, 4, 0.5, 20000, 1000, ))
    runP405 = multiprocessing.Process(target=runP, args=(env405, 5, 4, 0.5, 20000, ))
    runV405 = multiprocessing.Process(target=runV, args=(env405, 5, 4, 0.5, 20000, ))
    runQ405.start()
    runP405.start()
    runV405.start()
    runQ408 = multiprocessing.Process(target=runQ, args=(env408, 5, 4, 0.8, 20000, 1000, ))
    runP408 = multiprocessing.Process(target=runP, args=(env408, 5, 4, 0.8, 20000, ))
    runV408 = multiprocessing.Process(target=runV, args=(env408, 5, 4, 0.8, 20000, ))
    runQ408.start()
    runP408.start()
    runV408.start()
    runQ805 = multiprocessing.Process(target=runQ, args=(env805, 5, 8, 0.5, 20000, 1000, ))
    runP805 = multiprocessing.Process(target=runP, args=(env805, 5, 8, 0.5, 20000, ))
    runV805 = multiprocessing.Process(target=runV, args=(env805, 5, 8, 0.5, 20000, ))
    runQ805.start()
    runP805.start()
    runV805.start()
    runQ808 = multiprocessing.Process(target=runQ, args=(env808, 5, 8, 0.8, 20000, 1000, ))
    runP808 = multiprocessing.Process(target=runP, args=(env808, 5, 8, 0.8, 20000, ))
    runV808 = multiprocessing.Process(target=runV, args=(env808, 5, 8, 0.8, 20000, ))
    runQ808.start()
    runP808.start()
    runV808.start()

    runQ405.join()
    runP405.join()
    runV405.join()
    runQ408.join()
    runP408.join()
    runV408.join()
    runQ805.join()
    runP805.join()
    runV805.join()
    runQ808.join()
    runP808.join()
    runV808.join()

if __name__ == '__main__':
    main()