from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from keras import backend as K

import random
import os
import pickle

from .common import create_beta_list, create_gamma_list_agent57, rescaling_inverse, rescaling
from .model import LstmType


class Actor():
    def __init__(self,
            input_shape,
            input_sequence,
            nb_actions,
            action_policy,
            batch_size,
            lstm_type,
            reward_multisteps,
            lstmful_input_length,
            burnin_length,
            enable_intrinsic_reward,
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
            beta_max,
            gamma0,
            gamma1,
            gamma2,
            ucb_epsilon,
            ucb_beta,
            ucb_window_size,
            model_builder,
            actor_index,
        ):
        self.training = False

        self.input_shape = input_shape
        self.input_sequence = input_sequence
        self.nb_actions = nb_actions
        self.action_policy = action_policy
        self.batch_size = batch_size
        self.lstm_type = lstm_type
        self.reward_multisteps = reward_multisteps
        self.lstmful_input_length = lstmful_input_length
        self.burnin_length = burnin_length
        self.enable_intrinsic_reward = enable_intrinsic_reward
        self.enable_rescaling = enable_rescaling
        self.priority_exponent = priority_exponent
        self.int_episode_reward_k = int_episode_reward_k
        self.int_episode_reward_epsilon = int_episode_reward_epsilon
        self.int_episode_reward_c = int_episode_reward_c
        self.int_episode_reward_max_similarity = int_episode_reward_max_similarity
        self.int_episode_reward_cluster_distance = int_episode_reward_cluster_distance
        self.int_episodic_memory_capacity = int_episodic_memory_capacity
        self.rnd_err_capacity = rnd_err_capacity
        self.rnd_max_reward = rnd_max_reward
        self.policy_num = policy_num
        self.beta_max = beta_max

        self.ucb_epsilon = ucb_epsilon
        self.ucb_beta = ucb_beta
        self.ucb_window_size = ucb_window_size

        self.model_builder = model_builder
        
        #--- check
        if lstm_type == LstmType.STATEFUL:
            self.burnin_length = burnin_length
        else:
            self.burnin_length = 0

        assert reward_multisteps > 0, "'reward_multisteps' is 1 or more."
        
        # local val
        self.episode_count = 0
        self.policy_index = 0
        self.step = 0

        if self.enable_intrinsic_reward:
            self.rnd_err_vals = []
            self.int_beta_list = create_beta_list(policy_num, beta_max)
            self.gamma_list = create_gamma_list_agent57(policy_num, gamma0, gamma1, gamma2)
            self.ucb_data = []
        else:
            self.gamma_list = [gamma2]

        self.actor_index = actor_index


    def build_model(self, learner):
        if learner is None:
            self.actval_ext_model = self.model_builder.build_actval_func_model(None, enable_uvfa=False)
            if self.enable_intrinsic_reward:
                self.actval_int_model =  self.model_builder.build_actval_func_model(None, enable_uvfa=True)
                self.emb_model = self.model_builder.build_embedding_model()
                self.rnd_target_model = self.model_builder.build_rnd_model(None)
                self.rnd_train_model = self.model_builder.build_rnd_model(None)

        else:
            self.actval_ext_model = learner.actval_ext_model
            if self.enable_intrinsic_reward:
                self.actval_int_model = learner.actval_int_model
                self.emb_model = learner.emb_model
                self.rnd_target_model = learner.rnd_target_model
                self.rnd_train_model = learner.rnd_train_model

        if self.lstm_type == LstmType.STATEFUL:
            self.lstm = self.actval_ext_model.get_layer("lstm")
            if self.enable_intrinsic_reward:
                self.lstm_int = self.actval_int_model.get_layer("lstm")

    
    def load_weights(self, filepath):
        if not os.path.isfile(filepath):
            return
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.actval_ext_model.set_weights(d["weights_ext"])
        if self.enable_intrinsic_reward:
            self.actval_int_model.set_weights(d["weights_int"])
            self.rnd_train_model.set_weights(d["weights_rnd_train"])
            self.rnd_target_model.set_weights(d["weights_rnd_target"])
            self.emb_model.set_weights(d["weights_emb"])


    def episode_begin(self):
        self.recent_terminal = False
        self.total_reward = 0

        if self.lstm_type != LstmType.STATEFUL:
            self.recent_actions = [ 0 for _ in range(self.reward_multisteps+1)]
            self.recent_rewards = [ 0 for _ in range(self.reward_multisteps)]
            self.recent_rewards_multistep = 0
            self.recent_observations = [
                np.zeros(self.input_shape) for _ in range(self.input_sequence + self.reward_multisteps)
            ]
            self.recent_reward_intrinsic = 0
        else:
            multi_len = self.reward_multisteps + self.lstmful_input_length - 1
            self.recent_actions = [ 0 for _ in range(multi_len + 1)]
            self.recent_rewards = [ 0 for _ in range(multi_len)]
            self.recent_rewards_multistep = [ 0 for _ in range(self.lstmful_input_length)]
            tmp = self.burnin_length + self.input_sequence + multi_len
            self.recent_observations = [
                np.zeros(self.input_shape) for _ in range(tmp)
            ]
            tmp = self.burnin_length + multi_len + 1
            self.recent_observations_wrap = [
                [np.zeros(self.input_shape) for _ in range(self.input_sequence)] for _ in range(tmp)
            ]
            self.recent_reward_intrinsic = [ 0 for _ in range(self.lstmful_input_length)]

            # hidden_state: [(batch_size, lstm_units_num), (batch_size, lstm_units_num)]
            tmp = self.burnin_length + multi_len + 1+1
            self.actval_ext_model.reset_states()
            self.recent_hidden_states = [
                [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])] for _ in range(tmp)
            ]
            if self.enable_intrinsic_reward:
                self.actval_int_model.reset_states()
                self.recent_hidden_states_int = [
                    [K.get_value(self.lstm_int.states[0]), K.get_value(self.lstm_int.states[1])] for _ in range(tmp)
                ]
            else:
                self.recent_hidden_states_int = [0]


        if self.enable_intrinsic_reward:
            self.int_episodic_memory = []
            
            if not self.training:
                # traningじゃない場合はpolicy0(=探索なし)
                self.policy_index = 0
            else:
                # UCB計算用に保存
                if self.episode_count > 0:
                    self.ucb_data.append([
                        self.policy_index,
                        self.total_reward,
                    ])
                    if len(self.ucb_data) >= self.ucb_window_size:
                        self.ucb_data.pop(0)
                
                if self.episode_count < self.policy_num:
                    # 全て１回は実行
                    self.policy_index = self.episode_count
                else:
                    r = random.random()
                    if r < self.ucb_epsilon:
                        # ランダムでpolicyを決定
                        self.policy_index = random.randint(0, self.policy_num-1)  # a <= n <= b
                    else:
                        N = [1 for _ in range(self.policy_num)]
                        u = [0 for _ in range(self.policy_num)]
                        for d in self.ucb_data:
                            N[d[0]] += 1
                            u[d[0]] += d[1]

                        for i in range(self.policy_num):
                            u[i] /= N[i]
                        
                        count = len(self.ucb_data)
                        k = [0 for _ in range(self.policy_num)]
                        for i in range(self.policy_num):
                            k[i] = u[i] + self.ucb_beta * np.sqrt(np.log(count)/N[i])

                        self.policy_index =  np.argmax(k)
                
            self.episode_count += 1
        self.gamma = self.gamma_list[self.policy_index]
        

    def forward_train_before(self, observation):
        # observation
        self.recent_observations.pop(0)
        self.recent_observations.append(observation)

        if self.lstm_type == LstmType.STATEFUL:
            self.recent_observations_wrap.pop(0)
            self.recent_observations_wrap.append(self.recent_observations[-self.input_sequence:])
    
        # tmp
        self._qvals = None
        self._state_x = np.asarray(self.recent_observations[-self.input_sequence:])
        if self.lstm_type != LstmType.STATEFUL:
            # (1, input_shape)
            self._state_x = self._state_x[np.newaxis,:]
        else:
            # (batch_size, input_shape)
            self._state_x = np.full((self.batch_size,)+self._state_x.shape, self._state_x)

        if self.enable_intrinsic_reward:
            #t = to_categorical(self.recent_actions[-1], num_classes=self.nb_actions)
            t = to_categorical(self.policy_index, num_classes=self.policy_num)
            #t = np.append(t, self.recent_rewards[-1])
            #t = np.append(t, self.intrinsic_reward)
            if self.lstm_type == LstmType.NONE:
                # (1, policy_num)
                t = t[np.newaxis,:]
            elif self.lstm_type == LstmType.STATELESS:
                # (1, input_sequence, policy_num)
                t = np.full((self.input_sequence, self.policy_num) ,t)[np.newaxis,:]
            else:
                # (batch_size, input_sequence, policy_num)
                t = np.full((self.input_sequence, self.policy_num) ,t)
                t = np.full((self.batch_size,) + t.shape, t)
            self._state_x_int = [self._state_x, t]

    def forward_train_after(self):

        # hidden_state update
        if self.lstm_type == LstmType.STATEFUL:
            # hidden_state を更新しつつQ値も取得
            self.lstm.reset_states(self.recent_hidden_states[-1])
            self._qvals = self.actval_ext_model.predict(self._state_x, batch_size=self.batch_size)[0]
            hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
            self.recent_hidden_states.pop(0)
            self.recent_hidden_states.append(hidden_state)

            if self.enable_intrinsic_reward:
                self.lstm_int.reset_states(self.recent_hidden_states_int[-1])
                q_int_list = self.actval_int_model.predict(self._state_x_int, batch_size=self.batch_size)[0]
                hidden_state = [K.get_value(self.lstm_int.states[0]), K.get_value(self.lstm_int.states[1])]
                self.recent_hidden_states_int.pop(0)
                self.recent_hidden_states_int.append(hidden_state)

                # qvalの計算
                beta = self.int_beta_list[self.policy_index]
                for i in range(self.nb_actions):
                    q_ext = rescaling_inverse(self._qvals[i])
                    q_int = rescaling_inverse(q_int_list[i])
                    self._qvals[i] = rescaling(q_ext + beta * q_int)

            elif self.enable_rescaling:
                for i in range(self.nb_actions):
                    self._qvals[i] = rescaling(rescaling_inverse(self._qvals[i]))
        

        if self.recent_terminal:
            return 0  # 終了時はactionを出す必要がない

        if not self.training:
            # テスト中はQ値最大
            action = np.argmax(self.get_qvals())
        else:
            action = self.action_policy.select_action(self)

        # アクション保存
        self.recent_actions.pop(0)
        self.recent_actions.append(action)

        return action


    def get_qvals(self):
        if self._qvals is not None:
            # STATEFUL は hidden_state 計算時に計算済み
            return self._qvals

        self._qvals = self.actval_ext_model.predict(self._state_x, batch_size=1)[0]
        
        if self.enable_intrinsic_reward:
            q_int_list = self.actval_int_model.predict(self._state_x_int, batch_size=1)[0]
            beta = self.int_beta_list[self.policy_index]

            for i in range(self.nb_actions):
                q_ext = rescaling_inverse(self._qvals[i])
                q_int = rescaling_inverse(q_int_list[i])
                self._qvals[i] = rescaling(q_ext + beta * q_int)
        elif self.enable_rescaling:
            for i in range(self.nb_actions):
                self._qvals[i] = rescaling(rescaling_inverse(self._qvals[i]))
        
        return self._qvals


    def backward(self, reward, terminal):
        self.step += 1

        # terminal は env が終了状態ならTrue
        if not self.training:
            return []

        # 報酬の保存
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)
        self.total_reward += reward

        # 内部報酬を計算
        if self.enable_intrinsic_reward:
            self.int_episode_reward = self.calc_int_episode_reward()
            self.int_lifelong_reward = self.calc_int_lifelong_reward()
            _tmp = self.int_episode_reward * self.int_lifelong_reward
            if self.lstm_type != LstmType.STATEFUL:
                self.recent_reward_intrinsic = _tmp
            else:
                self.recent_reward_intrinsic.pop(0)
                self.recent_reward_intrinsic.append(_tmp)

        # multi step learning
        _tmp = 0
        for i in range(-self.reward_multisteps, 0):
            _tmp += self.recent_rewards[i] * (self.gamma ** (-i))
        if self.lstm_type != LstmType.STATEFUL:
            self.recent_rewards_multistep = _tmp
        else:
            self.recent_rewards_multistep.pop(0)
            self.recent_rewards_multistep.append(_tmp)
        
        # terminal
        self.recent_terminal = terminal
        return []


    def create_exp(self, calc_priority):

        if not calc_priority:
            priority = 0
        else:
            # priority の計算
            if self.lstm_type != LstmType.STATEFUL:
                state0 = self.recent_observations[-self.input_sequence-self.reward_multisteps:-self.reward_multisteps]
                state0 = np.asarray(state0)[np.newaxis,:]
                state1 = self._state_x
                action = self.recent_actions[0]
                reward = self.recent_rewards_multistep
                
                state0_qvals = self.actval_ext_model.predict(state0, 1)[0]
                state1_qvals = self.actval_ext_model.predict(state1, 1)[0]
                maxq = np.max(state1_qvals)
                td_error = reward + (self.gamma ** self.reward_multisteps) * maxq - state0_qvals[action]
                priority = abs(td_error)

            else:
                # 初回しか使わないので計算量のかかるburn-inは省略
                # (直前のhidden_statesなのでmodelによる誤差もほぼないため)

                prioritys = []
                for i in range(self.lstmful_input_length):
                    state0 = np.asarray(self.recent_observations_wrap[-self.reward_multisteps-1])
                    state0 = np.full((self.batch_size,)+state0.shape, state0)
                    state1 = self._state_x
                    hidden_states0 = self.recent_hidden_states[self.burnin_length + i]
                    hidden_states1 = self.recent_hidden_states[self.burnin_length + i + self.reward_multisteps]
                    action = self.recent_actions[i]
                    reward = self.recent_rewards_multistep[i]

                    # 現在のQネットワークを出力
                    self.lstm.reset_states(hidden_states0)
                    state0_qvals = self.actval_ext_model.predict(state0, self.batch_size)[0]
                    self.lstm.reset_states(hidden_states1)
                    state1_qvals = self.actval_ext_model.predict(state1, self.batch_size)[0]

                    maxq = np.max(state1_qvals)
                    td_error = reward + (self.gamma ** self.reward_multisteps) * maxq
                    priority = abs(td_error - state0_qvals[action])
                    prioritys.append(priority)
                
                # 今回使用したsamplingのpriorityを更新
                priority = self.priority_exponent * np.max(prioritys) + (1-self.priority_exponent) * np.average(prioritys)
                

        if self.lstm_type != LstmType.STATEFUL:
            exp = (
                self.recent_observations[:],
                self.recent_actions[0],
                self.recent_rewards_multistep,
                self.recent_rewards[-1],
                self.recent_terminal,
                priority,
                self.recent_reward_intrinsic,
                self.policy_index,
                self.actor_index,
            )
        else:
            exp = (
                self.recent_observations_wrap[:],
                self.recent_actions[:],
                self.recent_rewards_multistep[:],
                self.recent_rewards[-1],
                self.recent_terminal,
                priority,
                self.recent_reward_intrinsic[:],
                self.policy_index,
                self.actor_index,
                self.recent_hidden_states[0],
                self.recent_hidden_states_int[0]
            )
        return exp

    
    def calc_int_episode_reward(self):
        cont_state = self.emb_model.predict(self._state_x, batch_size=1)[0]

        if len(self.int_episodic_memory) == 0:
            episode_reward = 0
        else:
            # 全要素とのユークリッド距離を求める
            euclidean_list = [ np.linalg.norm(x - cont_state) for x in self.int_episodic_memory]

            # 上位k個が対象
            euclidean_list.sort()
            euclidean_list = euclidean_list[:self.int_episode_reward_k]
            euclidean_list = np.asarray(euclidean_list) ** 2

            # 平均
            ave = np.average(euclidean_list)
            if ave == 0:
                ave = 1

            # 訪問回数を近似
            count = 0
            for euclidean in euclidean_list:
                d = euclidean / ave
                d -= self.int_episode_reward_cluster_distance
                if d < euclidean_list[0]:
                    d = euclidean_list[0]
                count +=  self.int_episode_reward_epsilon / (d + self.int_episode_reward_epsilon)
            s = np.sqrt(count) + self.int_episode_reward_c

            if s > self.int_episode_reward_max_similarity:
                episode_reward = 0
            else:
                episode_reward = 1/s

        # エピソードメモリに追加
        self.int_episodic_memory.append(cont_state)
        if len(self.int_episodic_memory) >= self.int_episodic_memory_capacity:
            self.int_episodic_memory.pop(0)

        return episode_reward

    
    def calc_int_lifelong_reward(self):
        # RND取得
        rnd_target_val = self.rnd_target_model.predict(self._state_x, batch_size=1)[0]
        rnd_train_val = self.rnd_train_model.predict(self._state_x, batch_size=1)[0]

        # MSE
        mse = np.square(rnd_target_val - rnd_train_val).mean()
        self.rnd_err_vals.append(mse)
        if len(self.rnd_err_vals) > self.rnd_err_capacity:
            self.rnd_err_vals.pop(0)

        # 標準偏差
        sd = np.std(self.rnd_err_vals)
        if sd == 0:
            return 1

        # 平均
        ave = np.average(self.rnd_err_vals)

        # life long reward
        lifelong_reward = 1 + (mse - ave)/sd

        if lifelong_reward < 1:
            lifelong_reward = 1
        if lifelong_reward > self.rnd_max_reward:
            lifelong_reward = self.rnd_max_reward

        return lifelong_reward

