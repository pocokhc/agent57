from keras.models import model_from_json
from keras.optimizers import Adam
from keras.utils import to_categorical
import numpy as np
from keras import backend as K

import random
import os
import pickle

from .model import LstmType
from .common import create_beta_list, create_gamma_list_agent57
from .memory import EpisodeMemory, MemoryFactory

class Learner():
    def __init__(self,
            batch_size,
            nb_actions,
            target_model_update_interval,
            enable_double_dqn,
            enable_intrinsic_reward,
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
            actor_num,
        ):
        

        self.burnin_length = burnin_length
        self.reward_multisteps = reward_multisteps
        self.lstmful_input_length = lstmful_input_length
        self.priority_exponent = priority_exponent
        self.input_sequence = input_sequence

        self.nb_actions = nb_actions
        self.model_builder = model_builder
        self.policy_num = policy_num
        self.actor_num = actor_num

        # optimizer
        if optimizer_ext is None:
            optimizer_ext = Adam(lr=0.0001)
        if optimizer_int is None:
            optimizer_int = Adam(lr=0.0001)
        if optimizer_rnd is None:
            optimizer_rnd = Adam(lr=0.0005, epsilon=0.0001)
        if optimizer_emb is None:
            optimizer_emb = Adam(lr=0.0005, epsilon=0.0001)

        # model create
        self.actval_ext_model = model_builder.build_actval_func_model(optimizer_ext, enable_uvfa=False)
        self.actval_ext_model_target = model_from_json(self.actval_ext_model.to_json())
        self.actval_int_model = model_builder.build_actval_func_model(optimizer_int, enable_uvfa=True)
        self.actval_int_model_target = model_from_json(self.actval_int_model.to_json())
        self.rnd_target_model = model_builder.build_rnd_model(None)
        self.rnd_train_model = model_builder.build_rnd_model(optimizer_rnd)
        self.emb_model = model_builder.build_embedding_model()
        self.emb_train_model = model_builder.build_embedding_model_train(optimizer_emb)
        self.model_builder.sync_embedding_model(self.emb_train_model, self.emb_model)

        if lstm_type == LstmType.STATEFUL:
            self.lstm = self.actval_ext_model.get_layer("lstm")
            self.target_lstm = self.actval_ext_model_target.get_layer("lstm")
            self.lstm_int = self.actval_int_model.get_layer("lstm")
            self.target_lstm_int = self.actval_int_model_target.get_layer("lstm")


        self.memory = MemoryFactory.create(memory, memory_kwargs)
        self.memory_warmup_size = memory_warmup_size
        assert self.memory.capacity > batch_size, "Memory capacity is small.(Larger than batch size)"
        assert memory_warmup_size > batch_size, "Warmup steps is few.(Larger than batch size)"

        self.batch_size = batch_size
        self.lstm_type = lstm_type
        self.target_model_update_interval = target_model_update_interval
        self.enable_intrinsic_reward = enable_intrinsic_reward
        self.enable_double_dqn = enable_double_dqn

        # demo replay
        if demo_memory == "":
            self.demo_memory = None
            self.demo_ratio_initial = 0
            self.demo_ratio_final = 0
            self.demo_ratio_steps = 0
        else:
            self.demo_memory = MemoryFactory.create(demo_memory, demo_memory_kwargs)
            self.demo_ratio_initial = demo_ratio_initial
            if demo_ratio_final is None:
                self.demo_ratio_final = self.demo_ratio_initial
            else:
                self.demo_ratio_final = demo_ratio_final
            self.demo_ratio_steps = (self.demo_ratio_initial - self.demo_ratio_final) / demo_ratio_steps

        # episode replay
        if episode_memory == "":
            self.episode_memory = None
            self.episode_ratio = 0
        else:
            episode_memory = MemoryFactory.create(episode_memory, episode_memory_kwargs)
            self.episode_memory = EpisodeMemory(episode_memory, episode_verbose)
            self.episode_ratio = episode_ratio
            self.episode_memory_exp_list = [ [] for _ in range(self.actor_num)]
            self.total_reward_list = [ 0 for _ in range(self.actor_num)]

        # intrinsic_reward
        if self.enable_intrinsic_reward:
            self.int_beta_list = create_beta_list(policy_num, beta_max)
            self.gamma_list = create_gamma_list_agent57(policy_num, gamma0, gamma1, gamma2)
        else:
            self.gamma_list = [gamma2]

        # local
        self.train_count = 0


    def save_weights(self, filepath, overwrite=False, save_memory=False):  # override
        if not (overwrite or not os.path.isfile(filepath)):
            return
        d = {
            "weights_ext": self.actval_ext_model.get_weights(),
            "step": self.train_count,
        }
        if self.enable_intrinsic_reward:
            d["weights_int"] = self.actval_int_model.get_weights()
            d["weights_rnd_train"] = self.rnd_train_model.get_weights()
            d["weights_rnd_target"] = self.rnd_target_model.get_weights()
            self.model_builder.sync_embedding_model(self.emb_train_model, self.emb_model)
            d["weights_emb"] = self.emb_model.get_weights()
            d["weights_emb_train"] = self.emb_train_model.get_weights()
        with open(filepath, 'wb') as f:
            pickle.dump(d, f)
        
        # memory
        if save_memory:
            d = {}
            d["replay"] = self.memory.get_memorys()
            if self.episode_memory is not None:
                d["episode"] = self.episode_memory.get_memorys()
            with open(filepath + ".mem", 'wb') as f:
                pickle.dump(d, f)


    def load_weights(self, filepath, load_memory=False):  # override
        if not os.path.isfile(filepath):
            return
        with open(filepath, 'rb') as f:
            d = pickle.load(f)
        self.actval_ext_model.set_weights(d["weights_ext"])
        self.actval_ext_model_target.set_weights(d["weights_ext"])
        self.train_count = d["step"]
        if self.enable_intrinsic_reward:
            self.actval_int_model.set_weights(d["weights_int"])
            self.actval_int_model_target.set_weights(d["weights_int"])
            self.rnd_train_model.set_weights(d["weights_rnd_train"])
            self.rnd_target_model.set_weights(d["weights_rnd_target"])
            self.emb_model.set_weights(d["weights_emb"])
            self.emb_train_model.set_weights(d["weights_emb_train"])

        # memory
        if load_memory:
            filepath = filepath + ".mem"
            if os.path.isfile(filepath):
                with open(filepath, 'rb') as f:
                    d = pickle.load(f)
                self.memory.set_memorys(d["replay"])
                if "episode" in d and self.episode_memory is not None:
                    self.episode_memory.set_memorys(d["episode"])


    def train(self):

        # RemoteMemory が一定数貯まるまで学習しない。
        if len(self.memory) <= self.memory_warmup_size:
            return
        self.train_count += 1
        
        # batch ratio
        batch_replay = 0
        batch_demo = 0
        batch_episode = 0

        if self.demo_memory is None or len(self.demo_memory) < self.batch_size:
            ratio_demo = 0
        else:
            ratio_demo = self.demo_ratio_initial - self.train_count * self.demo_ratio_steps
            if ratio_demo < self.demo_ratio_final:
                ratio_demo = self.demo_ratio_final
        if self.episode_memory is None or len(self.episode_memory) < self.batch_size:
            ratio_epi = 0
        else:
            ratio_epi = self.episode_ratio
        for _ in range(self.batch_size):
            r = random.random()
            if r < ratio_demo:
                batch_demo += 1
                continue
            r -= ratio_demo
            if r < ratio_epi:
                batch_episode += 1
            else:
                batch_replay += 1
        batch_replay = self.batch_size
        batch_demo = 0
        batch_episode = 0
        
        # memory から優先順位にもとづき状態を取得
        indexes = []
        batchs = []
        weights = []
        memory_types = []
        if batch_replay > 0:
            (i, b, w) = self.memory.sample(batch_replay, self.train_count)
            indexes.extend(i)
            batchs.extend(b)
            weights.extend(w)
            memory_types.extend([0 for _ in range(batch_replay)])
        if batch_demo > 0:
            (i, b, w) = self.demo_memory.sample(batch_demo, self.train_count)
            indexes.extend(i)
            batchs.extend(b)
            weights.extend(w)
            memory_types.extend([1 for _ in range(batch_demo)])
        if batch_episode > 0:
            (i, b, w) = self.episode_memory.sample(batch_episode, self.train_count)
            indexes.extend(i)
            batchs.extend(b)
            weights.extend(w)
            memory_types.extend([2 for _ in range(batch_episode)])
        
        # 学習
        if self.lstm_type != LstmType.STATEFUL:
            self.train_model(indexes, batchs, weights, memory_types)
        else:
            self.train_model_lstmful(indexes, batchs, weights, memory_types)

        # target networkの更新
        if self.train_count % self.target_model_update_interval == 0:
            self.actval_ext_model_target.set_weights(self.actval_ext_model.get_weights())
            if self.enable_intrinsic_reward:
                self.actval_int_model_target.set_weights(self.actval_int_model.get_weights())

    # ノーマルの学習
    def train_model(self, indexes, batchs, weights, memory_types):

        state0_batch = []
        state1_batch = []
        if self.enable_intrinsic_reward:
            state0_batch_int = []
            state1_batch_int = []
            emb_act_batch = []
        for i, batch in enumerate(batchs):
            state0_batch.append(batch[0][-self.input_sequence-self.reward_multisteps:-self.reward_multisteps])
            state1_batch.append(batch[0][-self.input_sequence:])
            if self.enable_intrinsic_reward:
                # (policy_num)
                t = to_categorical(batch[7], num_classes=self.policy_num)
                if self.lstm_type != LstmType.NONE:
                    # (input_sequence, policy_num)
                    t = np.full((self.input_sequence, self.policy_num), t)
                state0_batch_int.append(t)
                state1_batch_int.append(t)
                emb_act_batch.append(to_categorical(batch[1], num_classes=self.nb_actions))
        state0_batch = np.asarray(state0_batch)
        state1_batch = np.asarray(state1_batch)

        # 更新用に現在のQネットワークを出力(Q network)
        state0_qvals = self.actval_ext_model.predict(state0_batch, self.batch_size)

        if self.enable_intrinsic_reward:
            emb_act_batch = np.asarray(emb_act_batch)
            state0_batch_int = [state0_batch, np.asarray(state0_batch_int)]
            state1_batch_int = [state1_batch, np.asarray(state1_batch_int)]

            # emb network
            self.emb_train_model.train_on_batch([state0_batch, state1_batch], emb_act_batch)

            # rnd network
            rnd_target_val = self.rnd_target_model.predict(state1_batch, self.batch_size)
            self.rnd_train_model.train_on_batch(state1_batch, rnd_target_val)

            # qvals
            state0_qvals_int = self.actval_int_model.predict(state0_batch_int, self.batch_size)

        if self.enable_double_dqn:
            # TargetNetworkとQNetworkのQ値を出す
            state1_qvals_model = self.actval_ext_model.predict(state1_batch, self.batch_size)
            state1_qvals_target = self.actval_ext_model_target.predict(state1_batch, self.batch_size)
            if self.enable_intrinsic_reward:
                #state1_qvals_model_int = self.actval_int_model.predict(state1_batch_int, self.batch_size)
                state1_qvals_target_int = self.actval_int_model_target.predict(state1_batch_int, self.batch_size)
        else:
            # 次の状態のQ値を取得(target_network)
            state1_qvals_target = self.actval_ext_model_target.predict(state1_batch, self.batch_size)
            if self.enable_intrinsic_reward:
                state1_qvals_target_int = self.actval_int_model_target.predict(state1_batch_int, self.batch_size)

        for i in range(self.batch_size):
            if self.enable_double_dqn:
                max_action = state1_qvals_model[i].argmax()  # modelからアクションを出す
                maxq = state1_qvals_target[i][max_action]    # Q値はtarget_modelを使って出す
            else:
                maxq = state1_qvals_target[i].max()

            policy_index = batchs[i][7]
            gamma = self.gamma_list[policy_index]
            action = batchs[i][1]
            reward = batchs[i][2]
            q0 = state0_qvals[i][action]

            # Calc
            td_error = reward + (gamma ** self.reward_multisteps) * maxq - q0
            priority = abs(td_error)
            state0_qvals[i][action] += td_error * weights[i]

            if self.enable_intrinsic_reward:
                if self.enable_double_dqn:
                    #max_action = state1_qvals_model_int[i].argmax()  # modelからアクションを出す
                    maxq = state1_qvals_target_int[i][max_action]    # Q値はtarget_modelを使って出す
                else:
                    maxq = state1_qvals_target_int[i].max()
                
                reward = batchs[i][6]
                q0 = state0_qvals_int[i][action]

                # Calc
                td_error = reward + (gamma ** self.reward_multisteps) * maxq - q0
                state0_qvals_int[i][action] += td_error * weights[i]

            # priorityを更新
            if memory_types[i] == 0:
                self.memory.update(indexes[i], batchs[i], priority)
            elif memory_types[i] == 1:
                self.demo_memory.update(indexes[i], batchs[i], priority)
            elif memory_types[i] == 2:
                self.episode_memory.update(indexes[i], batchs[i], priority)
            else:
                assert False

        # 学習
        self.actval_ext_model.train_on_batch(state0_batch, state0_qvals)
        if self.enable_intrinsic_reward:
            self.actval_int_model.train_on_batch(state0_batch_int, state0_qvals_int)


    # ステートフルLSTMの学習
    def train_model_lstmful(self, indexes, batchs, weights, memory_types):

        hidden_s0 = []
        hidden_s1 = []
        for batch in batchs:
            # batchサイズ分あるけどすべて同じなので0番目を取得
            hidden_s0.append(batch[9][0][0])
            hidden_s1.append(batch[9][1][0])
        hidden_states = [np.asarray(hidden_s0), np.asarray(hidden_s1)]

        # init hidden_state
        self.lstm.reset_states(hidden_states)
        self.target_lstm.reset_states(hidden_states)

        hidden_states_arr = []
        if self.burnin_length == 0:
            hidden_states_arr.append(hidden_states)
        state_batch_arr = []
        qvals_arr = []
        target_qvals_arr = []
        prioritys = [ [] for _ in range(self.batch_size)]

        if self.enable_intrinsic_reward:
            hidden_s0_int = []
            hidden_s1_int = []
            for batch in batchs:
                hidden_s0_int.append(batch[10][0][0])
                hidden_s1_int.append(batch[10][1][0])
            hidden_states_int = [np.asarray(hidden_s0_int), np.asarray(hidden_s1_int)]

            self.lstm_int.reset_states(hidden_states_int)
            self.target_lstm_int.reset_states(hidden_states_int)

            hidden_states_arr_int = []
            if self.burnin_length == 0:
                hidden_states_arr_int.append(hidden_states_int)
            state_batch_arr_int = []
            qvals_arr_int = []
            target_qvals_arr_int = []

        # predict
        for seq_i in range(self.burnin_length + self.reward_multisteps + self.lstmful_input_length):

            # state
            state_batch = [ batch[0][seq_i] for batch in batchs ]
            state_batch = np.asarray(state_batch)
            
            # hidden_state更新およびQ値取得
            qvals = self.actval_ext_model.predict(state_batch, self.batch_size)
            qvals_t = self.actval_ext_model_target.predict(state_batch, self.batch_size)

            if self.enable_intrinsic_reward:
                # (batch_size, input_sequence, policy_num)
                t = [ np.full((self.input_sequence, self.policy_num), 
                        to_categorical(batch[7], num_classes=self.policy_num)) for batch in batchs ]
                state_batch_int = [state_batch, np.asarray(t)]
                qvals_int = self.actval_int_model.predict(state_batch_int, self.batch_size)
                qvals_t_int = self.actval_int_model_target.predict(state_batch_int, self.batch_size)

            # burnin-1
            if seq_i < self.burnin_length-1:
                continue
            hidden_states_arr.append([K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])])

            if self.enable_intrinsic_reward:
                hidden_states_arr_int.append([K.get_value(self.lstm.states[0]), K.get_value(self.lstm.states[1])])

            # burnin
            if seq_i < self.burnin_length:
                continue

            state_batch_arr.append(state_batch)
            qvals_arr.append(qvals)
            target_qvals_arr.append(qvals_t)

            if self.enable_intrinsic_reward:
                state_batch_arr_int.append(state_batch_int)
                qvals_arr_int.append(qvals_int)
                target_qvals_arr_int.append(qvals_t_int)

        # train
        for seq_i in range(self.lstmful_input_length):

            # state0 の Qval (multistep前)
            state0_qvals = qvals_arr[seq_i]
            if self.enable_intrinsic_reward:
                state0_qvals_int = qvals_arr_int[seq_i]
            
            # batch
            for batch_i in range(self.batch_size):

                # maxq
                if self.enable_double_dqn:
                    max_action = qvals_arr[seq_i+self.reward_multisteps][batch_i].argmax()  # modelからアクションを出す
                    maxq = target_qvals_arr[seq_i+self.reward_multisteps][batch_i][max_action]  # Q値はtarget_modelを使って出す
                else:
                    maxq = target_qvals_arr[seq_i+self.reward_multisteps][batch_i].max()

                policy_index = batchs[batch_i][7]
                gamma = self.gamma_list[policy_index]
                action = batchs[batch_i][1][seq_i]
                reward = batchs[batch_i][2][seq_i]
                q0 = state0_qvals[batch_i][action]

                # Calc
                td_error = reward + (gamma ** self.reward_multisteps) * maxq - q0
                priority = abs(td_error)
                prioritys[batch_i].append(priority)
                state0_qvals[batch_i][action] += td_error * weights[batch_i]

                if self.enable_intrinsic_reward:
                    if self.enable_double_dqn:
                        max_action = qvals_arr_int[seq_i+self.reward_multisteps][batch_i].argmax()
                        maxq = target_qvals_arr_int[seq_i+self.reward_multisteps][batch_i][max_action]
                    else:
                        maxq = target_qvals_arr_int[seq_i+self.reward_multisteps][batch_i].max()

                    reward = batchs[batch_i][6][seq_i]
                    q0 = state0_qvals_int[batch_i][action]

                    # Calc
                    td_error = reward + (gamma ** self.reward_multisteps) * maxq - q0
                    state0_qvals_int[batch_i][action] += td_error * weights[batch_i]

            # train
            self.lstm.reset_states(hidden_states_arr[seq_i])
            self.actval_ext_model.train_on_batch(state_batch_arr[seq_i], state0_qvals)
            if self.enable_intrinsic_reward:
                self.lstm_int.reset_states(hidden_states_arr_int[seq_i])
                self.actval_int_model.train_on_batch(state_batch_arr_int[seq_i], state0_qvals_int)


        # priority update
        for i, batch in enumerate(batchs):
            priority = self.priority_exponent * np.max(prioritys[i]) + \
                (1-self.priority_exponent) * np.average(prioritys[i])

            # priorityを更新
            if memory_types[i] == 0:
                self.memory.update(indexes[i], batch, priority)
            elif memory_types[i] == 1:
                self.demo_memory.update(indexes[i], batch, priority)
            elif memory_types[i] == 2:
                self.episode_memory.update(indexes[i], batch, priority)
            else:
                assert False

        # 分かりやすさで別に記載
        if self.enable_intrinsic_reward:
            # embとrndの更新は最後の5fのみ使用する
            start = self.lstmful_input_length-5
            if start < 0:
                start = 0
            for i in range(start, self.lstmful_input_length):
                state0 = state_batch_arr[i]
                state1 = state_batch_arr[i+self.reward_multisteps]

                emb_act_batch = [ to_categorical(batch[1][i], num_classes=self.nb_actions) for batch in batchs ]
                emb_act_batch = np.asarray(emb_act_batch)

                # emb network
                self.emb_train_model.train_on_batch([state0, state1], emb_act_batch)

                # rnd network
                rnd_target_val = self.rnd_target_model.predict(state1, self.batch_size)
                self.rnd_train_model.train_on_batch(state1, rnd_target_val)




    def add_exp(self, exp):
        reward = exp[3]
        terminal = exp[4]
        priotiry = exp[5]
        actor_index = exp[8]

        # add memory
        if priotiry == 0:
            self.memory.add(exp)
        else:
            self.memory.add(exp, priotiry)

        # episode_memory
        if self.episode_memory is not None:
            self.episode_memory_exp_list[actor_index].append(exp)
            self.total_reward_list[actor_index] += reward

        # terminal
        if terminal:
            # episode_memory
            if self.episode_memory is not None:
                self.episode_memory.add_episode(
                    self.episode_memory_exp_list[actor_index],
                    self.total_reward_list[actor_index]
                )
                self.episode_memory_exp_list[actor_index] = []
                self.total_reward_list[actor_index] = 0

            # sync emb
            if self.enable_intrinsic_reward:
                self.model_builder.sync_embedding_model(self.emb_train_model, self.emb_model)

