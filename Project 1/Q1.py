import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # 使用TkAgg后端，适用于有图形界面的环境
from model import Model, Actions
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决坐标轴负号显示问题

def policy_evaluation(pi, model, V, gamma=0.9, tol=1e-6):
    while True:
        delta = 0
        for s in model.states:
            if s == model.fictional_end_state:
                continue
            v = V[s]
            a = pi[s]
            V[s] = np.sum([
                model.transition_probability(s, s_, a) * (model.reward(s, a) + gamma * V[s_])
                for s_ in model.states
            ])
            delta = max(delta, abs(v - V[s]))
        if delta < tol:
            break
    return V

def policy_iteration(model: Model, gamma: float = 0.9, tol: float = 1e-6):
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,), dtype=int)
    stable = False
    while not stable:
        V = policy_evaluation(pi, model, V, gamma, tol)
        stable = True
        for s in model.states:
            if s == model.fictional_end_state:
                continue
            old_action = pi[s]
            action_values = [
                np.sum([
                    model.transition_probability(s, s_, a) * (model.reward(s, a) + gamma * V[s_])
                    for s_ in model.states
                ]) for a in Actions
            ]
            pi[s] = Actions(np.argmax(action_values))
            if old_action != pi[s]:
                stable = False
    return V, pi

def value_iteration(model: Model, maxit: int = 100, tol: float = 1e-6):
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    for i in range(maxit):
        delta = 0
        for s in model.states:
            if s == model.fictional_end_state:
                continue
            v = V[s]
            action_values = [
                np.sum([
                    model.transition_probability(s, s_, a) * (model.reward(s, a) + model.gamma * V[s_])
                    for s_ in model.states
                ]) for a in Actions
            ]
            V[s] = max(action_values)
            pi[s] = Actions(np.argmax(action_values))
            delta = max(delta, abs(v - V[s]))
        if delta < tol:
            print(f"Value iteration converged after {i+1} iterations.")
            break
    return V, pi

def asynchronous_value_iteration(model: Model, maxit: int = 100, tol: float = 1e-6):
    V = np.zeros((model.num_states,))
    pi = np.zeros((model.num_states,))
    for i in range(maxit):
        delta = 0
        for s in model.states:
            if s == model.fictional_end_state:
                continue
            v = V[s]
            action_values = [
                np.sum([
                    model.transition_probability(s, s_, a) * (model.reward(s, a) + model.gamma * V[s_])
                    for s_ in model.states
                ]) for a in Actions
            ]
            V[s] = max(action_values)
            pi[s] = Actions(np.argmax(action_values))
            delta = max(delta, abs(v - V[s]))
            if delta < tol:
                break
        if delta < tol:
            print(f"Asynchronous value iteration converged after {i+1} iterations.")
            break
    return V, pi

if __name__ == "__main__":
    from world_config import cliff_world, small_world, grid_world
    from plot_vp import plot_vp

    model = Model(cliff_world)

    # 策略迭代
    V_pi, pi_pi = policy_iteration(model)
    plot_vp(model, V_pi, pi_pi)
    plt.title("Policy Iteration.")
    plt.show()

    # 值迭代
    V_vi, pi_vi = value_iteration(model)
    plot_vp(model, V_vi, pi_vi)
    plt.title("Value Iteration")
    plt.show()

    # 异步值迭代
    V_avi, pi_avi = asynchronous_value_iteration(model)
    plot_vp(model, V_avi, pi_avi)
    plt.title("Asynchronous Value Iteration")
    plt.show()

    print("Whether the policies obtained by the policy iteration and value iteration algorithms are identical:", np.all(pi_pi == pi_vi))
    print("Check whether the policies obtained by the policy iteration and asynchronous value iteration algorithms are exactly the same:", np.all(pi_pi == pi_avi))