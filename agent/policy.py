from keras.models import model_from_json
from keras.optimizers import Adam
import numpy as np

import random
import math



class Policy():
    """ Abstract base class for all implemented Policy. """
    def select_action(self, agent):
        raise NotImplementedError()
    

class EpsilonGreedy(Policy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, agent):
        if self.epsilon > random.random():
            # アクションをランダムに選択
            action = random.randint(0, agent.nb_actions-1)
        else:
            # 評価が最大のアクションを選択
            action = np.argmax(agent.get_qvals())
        return action


class EpsilonGreedyActor(EpsilonGreedy):
    def __init__(self, actor_index, actors_length, epsilon=0.4, alpha=7):
        if actors_length <= 1:
            tmp = epsilon ** (1 + alpha)
        else:
            tmp = epsilon ** (1 + actor_index/(actors_length-1)*alpha)
        super().__init__(epsilon=tmp)


class AnnealingEpsilonGreedy(Policy):
    """ native dqn pilocy
    https://arxiv.org/abs/1312.5602
    """

    def __init__(self,  
            initial_epsilon=1,  # 初期ε
            final_epsilon=0.1,  # 最終状態でのε
            exploration_steps=1_000_000  # 初期→最終状態になるまでのステップ数
        ):
        self.epsilon_step = (initial_epsilon - final_epsilon) / exploration_steps
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon

    def select_action(self, agent):

        # epsilon の計算
        epsilon = self.initial_epsilon - agent.step * self.epsilon_step
        if epsilon < self.final_epsilon:
            epsilon = self.final_epsilon

        if epsilon > random.random():
            # アクションをランダムに選択
            action = random.randint(0, agent.nb_actions-1)
        else:
            # 評価が最大のアクションを選択
            action = np.argmax(agent.get_qvals())
        return action


class SoftmaxPolicy(Policy):

    def select_action(self, agent):
        qvals = agent.get_qvals()
        exp_x = np.exp(qvals)

        vals = []
        for i in range(agent.nb_actions):
            # softmax 値以下の乱数を生成
            vals.append( random.uniform(0, exp_x[i]) )

        # 乱数の結果一番大きいアクションを選択
        action = np.argmax(vals)
        return action

