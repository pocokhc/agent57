from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from keras import backend as K

import random
import os
import pickle

from .common import create_beta_list, create_gamma_list_agent57, rescaling_inverse, rescaling
from .model import LstmType, UvfaType


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
            actor_index,
        ):
        self.training = False
        self.test_policy = test_policy

        self.input_shape = input_shape
        self.input_sequence = input_sequence
        self.nb_actions = nb_actions
        self.action_policy = action_policy
        self.batch_size = batch_size
        self.lstm_type = lstm_type
        self.reward_multisteps = reward_multisteps
        self.lstmful_input_length = lstmful_input_length
        self.burnin_length = burnin_length
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
        self.uvfa_ext = uvfa_ext
        self.uvfa_int = uvfa_int
        
        self.enable_intrinsic_actval_model = enable_intrinsic_actval_model
        if self.enable_intrinsic_actval_model or (UvfaType.REWARD_INT in self.uvfa_ext):
            self.enable_intrinsic_reward = True
        else:
            self.enable_intrinsic_reward = False

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

        # ucb
        self.int_beta_list = create_beta_list(policy_num, beta_max)
        self.gamma_list = create_gamma_list_agent57(policy_num, gamma0, gamma1, gamma2)
        self.ucb_data = []

        self.actor_index = actor_index


    def build_model(self, learner):
        if learner is None:
            self.actval_ext_model = self.model_builder.build_actval_func_model(None, uvfa=self.uvfa_ext)
            if self.enable_intrinsic_actval_model:
                self.actval_int_model =  self.model_builder.build_actval_func_model(None, uvfa=self.uvfa_int)
            if self.enable_intrinsic_reward:
                self.emb_model = self.model_builder.build_embedding_model()
                self.rnd_target_model = self.model_builder.build_rnd_model(None)
                self.rnd_train_model = self.model_builder.build_rnd_model(None)

        else:
            self.actval_ext_model = learner.actval_ext_model
            if self.enable_intrinsic_actval_model:
                self.actval_int_model = learner.actval_int_model
            if self.enable_intrinsic_reward:
                self.emb_model = learner.emb_model
                self.rnd_target_model = learner.rnd_target_model
                self.rnd_train_model = learner.rnd_train_model

        if self.lstm_type == LstmType.STATEFUL:
            self.lstm = self.actval_ext_model.get_layer("lstm")
            if self.enable_intrinsic_actval_model:
                self.lstm_int = self.actval_int_model.get_layer("lstm")

    
    def load_weights(self, filepath):
        if not os.path.isfile(filepath):
            return
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.actval_ext_model.set_weights(d["weights_ext"])
        if self.enable_intrinsic_actval_model:
            self.actval_int_model.set_weights(d["weights_int"])
        if self.enable_intrinsic_reward:
            self.rnd_train_model.set_weights(d["weights_rnd_train"])
            self.rnd_target_model.set_weights(d["weights_rnd_target"])
            self.emb_model.set_weights(d["weights_emb"])


    def episode_begin(self):
        self.recent_terminal = False
        self.total_reward = 0
        self.backward_run = False

        if self.lstm_type != LstmType.STATEFUL:
            self.recent_actions = [ 0 for _ in range(self.reward_multisteps*2)]
            self.recent_rewards = [ 0 for _ in range(self.reward_multisteps)]
            self.recent_rewards_multistep = [ 0 for _ in range(self.reward_multisteps+1)]
            self.recent_observations = [
                np.zeros(self.input_shape) for _ in range(self.input_sequence + self.reward_multisteps)
            ]
            if self.enable_intrinsic_reward:
                self.recent_reward_intrinsic = [ 0 for _ in range(self.reward_multisteps)]
                self.recent_reward_intrinsic_multistep = [ 0 for _ in range(self.reward_multisteps+1)]
            else:
                self.recent_reward_intrinsic_multistep = [ 0 for _ in range(self.reward_multisteps+1)]

        else:
            obsnum = self.lstmful_input_length + self.burnin_length
            actlen = self.reward_multisteps*2 + obsnum - 1
            rwdlen = self.reward_multisteps
            rwdmultilen = self.reward_multisteps + obsnum
            obslen = self.input_sequence
            obswraplen = self.reward_multisteps + obsnum

            self.recent_actions = [ 0 for _ in range(actlen)]
            self.recent_rewards = [ 0 for _ in range(rwdlen)]
            self.recent_rewards_multistep = [ 0 for _ in range(rwdmultilen)]
            self.recent_observations = [ np.zeros(self.input_shape) for _ in range(obslen)]
            self.recent_observations_wrap = [ [np.zeros(self.input_shape) for _ in range(self.input_sequence)] for _ in range(obswraplen)]

            intlen = self.reward_multisteps
            intmultilen = self.reward_multisteps + obsnum
            if self.enable_intrinsic_reward:
                self.recent_reward_intrinsic = [ 0 for _ in range(intlen)]
                self.recent_reward_intrinsic_multistep = [ 0 for _ in range(intmultilen)]
            else:
                self.recent_reward_intrinsic_multistep = [ 0 for _ in range(intmultilen)]

            # hidden_state: [(batch_size, lstm_units_num), (batch_size, lstm_units_num)]
            hidlen = self.reward_multisteps + obsnum
            self.actval_ext_model.reset_states()
            hidden_zero = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
            self.recent_hidden_states = [ hidden_zero for _ in range(hidlen)]

            if self.enable_intrinsic_actval_model:
                self.actval_int_model.reset_states()
                hidden_zero = [K.get_value(self.lstm_int.states[0]), K.get_value(self.lstm_int.states[1])]
                self.recent_hidden_states_int = [ hidden_zero for _ in range(hidlen)]
            else:
                self.recent_hidden_states_int = [0]


        if self.enable_intrinsic_reward:
            self.int_episodic_memory = []
            
        if not self.training:
            self.policy_index = self.test_policy
        elif self.policy_num == 1:
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
                    # UCB
                    self.policy_index = self._get_ucb_policy()
                
            self.episode_count += 1
        self.gamma = self.gamma_list[self.policy_index]

    def _get_ucb_policy(self):
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

        return np.argmax(k)


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
        self._state_x_ext = None
        self._state_x_int = None
        if self.lstm_type != LstmType.STATEFUL:
            # (1, input_shape)
            self._state_x = self._state_x[np.newaxis,:]
        else:
            # (batch_size, input_shape)
            self._state_x = np.full((self.batch_size,)+self._state_x.shape, self._state_x)

        self._state_x_ext = self._create_uvfa_input(self._state_x, self.uvfa_ext)
        if self.enable_intrinsic_actval_model:
            self._state_x_int = self._create_uvfa_input(self._state_x, self.uvfa_int)

    def _create_uvfa_input(self, state, uvfa_types):
        if len(uvfa_types) == 0:
            return state
        else:
            t = np.empty(0)
            if UvfaType.ACTION in uvfa_types:
                t = np.append(t, to_categorical(self.recent_actions[self.reward_multisteps], num_classes=self.nb_actions))
            if UvfaType.REWARD_EXT in uvfa_types:
                t = np.append(t, self.recent_rewards_multistep[self.reward_multisteps])
            if UvfaType.REWARD_INT in uvfa_types:
                t = np.append(t, self.recent_reward_intrinsic_multistep[self.reward_multisteps])
            if UvfaType.POLICY in uvfa_types:
                t = np.append(t, to_categorical(self.policy_index, num_classes=self.policy_num))

            if self.lstm_type == LstmType.NONE:
                # (1, uvfa_input)
                t = t[np.newaxis,:]
            elif self.lstm_type == LstmType.STATELESS:
                # (1, input_sequence, uvfa_input)
                t = np.full((self.input_sequence,) + t.shape ,t)[np.newaxis,:]
            else:
                # (batch_size, input_sequence, policy_num)
                t = np.full((self.input_sequence,) + t.shape ,t)
                t = np.full((self.batch_size,) + t.shape, t)
            return [state, t]


    def forward_train_after(self):

        # hidden_state update
        if self.lstm_type == LstmType.STATEFUL:
            # hidden_state を更新しつつQ値も取得
            self.lstm.reset_states(self.recent_hidden_states[-1])
            self._qvals = self.actval_ext_model.predict(self._state_x_ext, batch_size=self.batch_size)[0]
            hidden_state = [K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])]
            self.recent_hidden_states.pop(0)
            self.recent_hidden_states.append(hidden_state)

            if self.enable_intrinsic_actval_model:
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

        if self.training:
            action = self.action_policy.select_action(self)
        else:
            # テスト中はQ値最大
            return np.argmax(self.get_qvals())

        # アクション保存
        self.recent_actions.pop(0)
        self.recent_actions.append(action)

        return action


    def get_qvals(self):
        if self._qvals is not None:
            # STATEFUL は hidden_state 計算時に計算済み
            return self._qvals

        self._qvals = self.actval_ext_model.predict(self._state_x_ext, batch_size=1)[0]
        
        if self.enable_intrinsic_actval_model:
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
        # terminal は env が終了状態ならTrue
        self.step += 1

        # terminal
        self.recent_terminal = terminal
        self.backward_run = True

        # reward
        self.recent_rewards.pop(0)
        self.recent_rewards.append(reward)
        self.total_reward += reward

        # multi step learning
        self.recent_rewards_multistep.pop(0)
        self.recent_rewards_multistep.append(self.calc_multistep_reward())
        
        # 内部報酬を計算
        if self.enable_intrinsic_reward:
            self.int_episode_reward = self.calc_int_episode_reward()
            self.int_lifelong_reward = self.calc_int_lifelong_reward()
            _tmp = self.int_episode_reward * self.int_lifelong_reward

            self.recent_reward_intrinsic.pop(0)
            self.recent_reward_intrinsic.append(_tmp)

            # multi step learning, 内部報酬は multistep すると学習しない？
            #_tmp = 0
            #for i in range(-self.reward_multisteps, 0):
            #    _tmp += self.recent_reward_intrinsic[i] * (self.gamma ** (self.reward_multisteps+i+1))
            self.recent_reward_intrinsic_multistep.pop(0)
            self.recent_reward_intrinsic_multistep.append(_tmp)

        
        return []


    def add_episode_end_frame(self):
        # 最終フレーム後に1フレーム追加
        self.recent_observations.pop(0)
        self.recent_observations.append(np.zeros(self.input_shape))
        #self.recent_actions.pop(0)
        #self.recent_actions.append(0)
        self.recent_rewards.pop(0)
        self.recent_rewards.append(0)
        self.recent_rewards_multistep.pop(0)
        self.recent_rewards_multistep.append(self.calc_multistep_reward())
        if self.enable_intrinsic_reward:
            self.recent_reward_intrinsic_multistep.pop(0)
            self.recent_reward_intrinsic_multistep.append(0)
        if self.lstm_type == LstmType.STATEFUL:
            self.recent_observations_wrap.pop(0)
            self.recent_observations_wrap.append(self.recent_observations[-self.input_sequence:])
            self.actval_ext_model.reset_states()
            self.recent_hidden_states.pop(0)
            self.recent_hidden_states.append([K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])])
            if self.enable_intrinsic_actval_model:
                self.actval_int_model.reset_states()
                self.recent_hidden_states_int.pop(0)
                self.recent_hidden_states_int.append([K.get_value(self.lstm_int.states[0]), K.get_value(self.lstm_int.states[1])])

    def create_exp(self, calc_priority, update_terminal=None):
        if not self.training:
            return None
        if not self.backward_run:
            return None

        if not calc_priority:
            priority = 0
        else:
            # priority の計算
            if self.lstm_type != LstmType.STATEFUL:
                state0 = self.recent_observations[-self.input_sequence-self.reward_multisteps:-self.reward_multisteps]
                state0 = np.asarray(state0)[np.newaxis,:]
                state1 = self._state_x
                action1 = self.recent_actions[self.reward_multisteps]
                reward1 = self.recent_rewards_multistep[self.reward_multisteps]

                if len(self.uvfa_ext) > 0:
                    ext0 = np.empty(0)
                    ext1 = np.empty(0)
                    if UvfaType.ACTION in self.uvfa_ext:
                        act0 = to_categorical(self.recent_actions[0], num_classes=self.nb_actions)
                        act1 = to_categorical(action1, num_classes=self.nb_actions)
                        ext0 = np.append(ext0, act0)
                        ext1 = np.append(ext1, act1)
                    if UvfaType.REWARD_EXT in self.uvfa_ext:
                        ext0 = np.append(ext0, self.recent_rewards_multistep[0])
                        ext1 = np.append(ext1, reward1)
                    if UvfaType.REWARD_INT in self.uvfa_ext:
                        ext0 = np.append(ext0, self.recent_reward_intrinsic_multistep[0])
                        ext1 = np.append(ext1, self.recent_reward_intrinsic_multistep[self.reward_multisteps])
                    if UvfaType.POLICY in self.uvfa_ext:
                        policy = to_categorical(self.policy_index, num_classes=self.policy_num)
                        ext0 = np.append(ext0, policy)
                        ext1 = np.append(ext1, policy)
                    
                    if self.lstm_type != LstmType.NONE:
                        # (input_sequence, uvfa_input)
                        ext0 = np.full((self.input_sequence,) + ext0.shape, ext0)
                        ext1 = np.full((self.input_sequence,) + ext1.shape, ext1)
                    
                    state0 = [state0, ext0[np.newaxis,:]]
                    state1 = [state1, ext1[np.newaxis,:]]
                
                state0_qvals = self.actval_ext_model.predict(state0, 1)[0]
                state1_qvals = self.actval_ext_model.predict(state1, 1)[0]
                maxq = np.max(state1_qvals)
                td_error = reward1 + (self.gamma ** self.reward_multisteps) * maxq - state0_qvals[action1]
                priority = abs(td_error)

            else:
                # 初回しか使わないので計算量のかかるburn-inは省略
                # (直前のhidden_statesなのでmodelによる誤差もほぼないため)
                prioritys = []
                if UvfaType.POLICY in self.uvfa_ext:
                    policy = to_categorical(self.policy_index, num_classes=self.policy_num)
                for i in range(self.lstmful_input_length):
                    state0 = np.asarray(self.recent_observations_wrap[-1-i-self.reward_multisteps])
                    state0 = np.full((self.batch_size,)+state0.shape, state0)
                    state1 = np.asarray(self.recent_observations_wrap[-1-i])
                    state1 = np.full((self.batch_size,)+state1.shape, state1)
                    hidden_states0 = self.recent_hidden_states[-1-i-self.reward_multisteps]
                    hidden_states1 = self.recent_hidden_states[-1-i]
                    action1 = self.recent_actions[-1-i-self.reward_multisteps+1]
                    reward1 = self.recent_rewards_multistep[-1-i]

                    if len(self.uvfa_ext) > 0:
                        ext0 = np.empty(0)
                        ext1 = np.empty(0)
                        if UvfaType.ACTION in self.uvfa_ext:
                            act0 = to_categorical(self.recent_actions[-1-i-self.reward_multisteps*2+1], num_classes=self.nb_actions)
                            act1 = to_categorical(action1, num_classes=self.nb_actions)
                            ext0 = np.append(ext0, act0)
                            ext1 = np.append(ext1, act1)
                        if UvfaType.REWARD_EXT in self.uvfa_ext:
                            ext0 = np.append(ext0, self.recent_rewards_multistep[-1-i-self.reward_multisteps])
                            ext1 = np.append(ext1, reward1)
                        if UvfaType.REWARD_INT in self.uvfa_ext:
                            ext0 = np.append(ext0, self.recent_reward_intrinsic_multistep[-1-i-self.reward_multisteps])
                            ext1 = np.append(ext1, self.recent_reward_intrinsic_multistep[-1-i])
                        if UvfaType.POLICY in self.uvfa_ext:
                            ext0 = np.append(ext0, policy)
                            ext1 = np.append(ext1, policy)
                        
                        # (batch_size, input_sequence, uvfa_input)
                        ext0 = np.full((self.input_sequence,) + ext0.shape, ext0)
                        ext1 = np.full((self.input_sequence,) + ext1.shape, ext1)
                        ext0 = np.full((self.batch_size,) + ext0.shape, ext0)
                        ext1 = np.full((self.batch_size,) + ext1.shape, ext1)
                        state0 = [state0, ext0]
                        state1 = [state1, ext1]

                    # 現在のQネットワークを出力
                    self.lstm.reset_states(hidden_states0)
                    state0_qvals = self.actval_ext_model.predict(state0, self.batch_size)[0]
                    self.lstm.reset_states(hidden_states1)
                    state1_qvals = self.actval_ext_model.predict(state1, self.batch_size)[0]

                    maxq = np.max(state1_qvals)
                    td_error = reward1 + (self.gamma ** self.reward_multisteps) * maxq
                    priority = abs(td_error - state0_qvals[action1])
                    prioritys.append(priority)
                
                # 今回使用したsamplingのpriorityを更新
                priority = self.priority_exponent * np.max(prioritys) + (1-self.priority_exponent) * np.average(prioritys)

        if update_terminal is None:
            terminal = self.recent_terminal
        else:
            terminal = update_terminal

        if self.lstm_type != LstmType.STATEFUL:
            exp = (
                self.recent_observations[:],
                [self.recent_actions[0], self.recent_actions[self.reward_multisteps]],
                [self.recent_rewards_multistep[0], self.recent_rewards_multistep[self.reward_multisteps]],
                self.recent_rewards[self.reward_multisteps-1],
                terminal,
                priority,
                [self.recent_reward_intrinsic_multistep[0], self.recent_reward_intrinsic_multistep[self.reward_multisteps]],
                self.policy_index,
                self.actor_index,
            )
        else:
            exp = (
                self.recent_observations_wrap[:],
                self.recent_actions[:],
                self.recent_rewards_multistep[:],
                self.recent_rewards[-1],
                terminal,
                priority,
                self.recent_reward_intrinsic_multistep[:],
                self.policy_index,
                self.actor_index,
                self.recent_hidden_states[0],
                self.recent_hidden_states_int[0]
            )
        return exp


    def calc_multistep_reward(self):
        _tmp = 0
        for i in range(-self.reward_multisteps, 0):
            _tmp += self.recent_rewards[i] * (self.gamma ** (self.reward_multisteps+i))
        return _tmp
    

    def calc_int_episode_reward(self):
        cont_state = self.emb_model.predict(self._state_x, batch_size=1)[0]

        if len(self.int_episodic_memory) == 0:
            episode_reward = 1
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

