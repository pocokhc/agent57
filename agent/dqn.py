
import rl
import rl.core
import keras
from keras.layers import Input, Flatten, Permute, TimeDistributed, LSTM, Dense, Concatenate, Reshape, Lambda
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
import numpy as np

from keras.optimizers import Adam
from keras.utils import to_categorical

import os
import pickle
import random
import time
import json
import matplotlib.pyplot as plt

from .model import ModelBuilder
from .model import DuelingNetwork, LstmType
from .actor import Actor
from .learner import Learner
from .env_play import add_memory

class DQN(rl.core.Agent):
    def __init__(self,
        input_shape,
        input_type,
        nb_actions,
        memory,
        memory_kwargs,
        action_policy,

        # input_model
        input_model=None,
        input_model_emb=None,
        input_model_rnd=None,

        # optimizer
        optimizer_ext=None,
        optimizer_int=None,
        optimizer_rnd=None,
        optimizer_emb=None,

        # NN
        batch_size=32,
        input_sequence=4,    # 入力フレーム数
        dense_units_num=512, # Dense層のユニット数
        enable_dueling_network=True,                  # dueling network有効フラグ
        dueling_network_type=DuelingNetwork.AVERAGE,  # dueling networkで使うアルゴリズム
        lstm_type=LstmType.NONE,  # LSTM有効フラグ
        lstm_units_num=512,       # LSTMのユニット数
        lstmful_input_length=80,  # ステートフルLSTMの入力数

        # train関係
        memory_warmup_size=50000, # 初期メモリー確保用step数(学習しない)
        target_model_update_interval=500,  # target networkのupdate間隔
        enable_double_dqn=True,   # DDQN有効フラグ
        enable_rescaling=False,   # rescalingを有効にするか
        rescaling_epsilon=0.001,  # rescalingの定数
        priority_exponent=0.9,    # シーケンス長priorityを計算する際のη
        burnin_length=4,          # burn-in期間
        reward_multisteps=3,      # multistep reward

        # demo memory
        demo_memory="",
        demo_memory_kwargs={},
        demo_episode_dir="",
        demo_ratio_initial=1.0/256.0,
        demo_ratio_final=None,
        demo_ratio_steps=100_000,

        # episode memory
        episode_memory="",
        episode_memory_kwargs={},
        episode_ratio=1.0/256.0,
        episode_verbose=1,

        # intrinsic_reward
        uvfa_ext=[],
        uvfa_int=[],
        enable_intrinsic_actval_model=False,
        int_episode_reward_k=10,
        int_episode_reward_epsilon=0.001,
        int_episode_reward_c=0.001,
        int_episode_reward_max_similarity=8,
        int_episode_reward_cluster_distance=0.008,
        int_episodic_memory_capacity=30000,
        rnd_err_capacity=10_000,
        rnd_max_reward=5,
        policy_num=1,
        beta_max=0.3,
        gamma0=0.9999,
        gamma1=0.997,
        gamma2=0.99,
        ucb_epsilon=0.5,
        ucb_beta=1,
        ucb_window_size=90,

        # other
        processor=None,
        step_interval=1,
        enable_add_episode_end_frame=True,
        test_policy=0,
    ):
        super().__init__(processor)
        self.compiled = True  # super

        self.step_interval = step_interval
        self.enable_add_episode_end_frame = enable_add_episode_end_frame

        if not enable_intrinsic_actval_model:
            uvfa_int = []

        # model
        model_builder = ModelBuilder(
            input_shape,
            input_type,
            input_model,
            input_model_emb,
            input_model_rnd,
            batch_size,
            nb_actions,
            input_sequence,
            enable_dueling_network,
            dueling_network_type,
            dense_units_num,
            lstm_type,
            lstm_units_num,
            policy_num,
        )

        # learner
        self.learner = Learner(
            batch_size,
            nb_actions,
            target_model_update_interval,
            enable_double_dqn,
            enable_intrinsic_actval_model,
            lstm_type,
            memory,
            memory_kwargs,
            memory_warmup_size,
            model_builder,
            optimizer_ext,
            optimizer_int,
            optimizer_rnd,
            optimizer_emb,
            demo_memory,
            demo_memory_kwargs,
            demo_ratio_initial,
            demo_ratio_steps,
            demo_ratio_final,
            episode_memory,
            episode_memory_kwargs,
            episode_ratio,
            episode_verbose,
            reward_multisteps,
            burnin_length,
            lstmful_input_length,
            priority_exponent,
            input_sequence,
            policy_num,
            beta_max,
            gamma0,
            gamma1,
            gamma2,
            uvfa_ext,
            uvfa_int,
            actor_num=1,
        )

        
        # demo memory
        if self.learner.demo_memory is not None:
            add_memory(demo_episode_dir, self.learner.demo_memory, model_builder, {
                "input_shape": input_shape,
                "input_sequence": input_sequence,
                "nb_actions": nb_actions,
                "batch_size": batch_size,
                "lstm_type": lstm_type,
                "reward_multisteps": reward_multisteps,
                "lstmful_input_length": lstmful_input_length,
                "burnin_length": burnin_length,
                "enable_intrinsic_actval_model": enable_intrinsic_actval_model,
                "enable_rescaling": enable_rescaling,
                "priority_exponent": priority_exponent,
                "int_episode_reward_k": int_episode_reward_k,
                "int_episode_reward_epsilon": int_episode_reward_epsilon,
                "int_episode_reward_c": int_episode_reward_c,
                "int_episode_reward_max_similarity": int_episode_reward_max_similarity,
                "int_episode_reward_cluster_distance": int_episode_reward_cluster_distance,
                "int_episodic_memory_capacity": int_episodic_memory_capacity,
                "rnd_err_capacity": rnd_err_capacity,
                "rnd_max_reward": rnd_max_reward,
                "policy_num": policy_num,
                "test_policy": test_policy,
                "beta_max": beta_max,
                "gamma0": gamma0,
                "gamma1": gamma1,
                "gamma2": gamma2,
                "ucb_epsilon": ucb_epsilon,
                "ucb_beta": ucb_beta,
                "ucb_window_size": ucb_window_size,
                "model_builder": model_builder,
                "uvfa_ext": uvfa_ext,
                "uvfa_int": uvfa_int,
                "step_interval": step_interval,
            })


        # actor
        self.actor = Actor(
            input_shape,
            input_sequence,
            nb_actions,
            action_policy,
            batch_size,
            lstm_type,
            reward_multisteps,
            lstmful_input_length,
            burnin_length,
            enable_intrinsic_actval_model,
            enable_rescaling,
            priority_exponent,
            int_episode_reward_k,
            int_episode_reward_epsilon,
            int_episode_reward_c,
            int_episode_reward_max_similarity,
            int_episode_reward_cluster_distance,
            int_episodic_memory_capacity,
            rnd_err_capacity,
            rnd_max_reward,
            policy_num,
            test_policy,
            beta_max,
            gamma0,
            gamma1,
            gamma2,
            ucb_epsilon,
            ucb_beta,
            ucb_window_size,
            model_builder,
            uvfa_ext,
            uvfa_int,
            actor_index=0,
        )

        # model share
        self.actor.build_model(self.learner)


    def reset_states(self):  # override
        self.actor.training = self.training  # training はコンストラクタで初期化されない
        self.actor.episode_begin()

        self.local_step = 0
        self.repeated_action = 0
        self.step_reward = 0
        self.recent_terminal = False
        
        
    def compile(self, optimizer, metrics=[]):  # override
        self.compiled = True  # super

    def save_weights(self, filepath, overwrite=False, save_memory=False):  # override
        self.learner.save_weights(filepath, overwrite, save_memory)

    def load_weights(self, filepath, load_memory=False):  # override
        self.learner.load_weights(filepath, load_memory)

    def forward(self, observation):  # override

        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.recent_terminal or (self.local_step % self.step_interval == 0):
            self.actor.forward_train_before(observation)

            if self.recent_terminal and self.enable_add_episode_end_frame:
                # 最終フレーム後に1フレーム追加
                exp = self.actor.create_exp(False, update_terminal=False)
                if exp is not None:
                    self.learner.add_exp(exp)
                self.actor.add_episode_end_frame()
                exp = self.actor.create_exp(False, update_terminal=True)
                if exp is not None:
                    self.learner.add_exp(exp)
                    self.learner.train()
            else:
                exp = self.actor.create_exp(False)
                if exp is not None:
                    self.learner.add_exp(exp)
                    self.learner.train()
            
            action = self.actor.forward_train_after()
            self.repeated_action = action

        return action

    
    def backward(self, reward, terminal):  # override
        self.step_reward += reward
        self.recent_terminal = terminal
        if terminal or (self.local_step % self.step_interval == 0):
            self.actor.backward(self.step_reward, terminal)
            self.step_reward = 0
        
        self.local_step += 1
        return []
    
    
    @property
    def layers(self):  #override
        return []
    

