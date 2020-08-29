import tensorflow as tf
import rl
import rl.core
import keras
from keras.layers import Input, Flatten, Permute, TimeDistributed, LSTM, Dense, Concatenate, Reshape, Lambda
from keras.models import Model
from keras.models import model_from_json
from keras import backend as K
import numpy as np

import multiprocessing as mp
import math
import os
import pickle
import enum
import time
import traceback
import ctypes
import random

from .model import ModelBuilder
from .model import DuelingNetwork, LstmType, UvfaType
from .actor import Actor
from .learner import Learner
from .env_play import add_memory


# 複数のプロセスでGPUを使用する設定
# https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
# https://github.com/tensorflow/tensorflow/issues/11812
# https://www.tensorflow.org/guide/gpu

# 最低でも1024MB以上、memory_limit * (actor数+2)がGPUメモリを超えない事
#memory_limit = 1024
memory_limit = 0
for device in tf.config.experimental.list_physical_devices('GPU'):
    if memory_limit == 0:
        tf.config.experimental.set_memory_growth(device, True)
    else:
        tf.config.experimental.set_virtual_device_configuration(device,
            [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)])


class Agent57():
    def __init__(self, 
        input_shape,
        input_type,
        nb_actions,
        memory,
        memory_kwargs,
        actors,

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
        policy_num=8,
        beta_max=0.3,
        gamma0=0.9999,
        gamma1=0.997,
        gamma2=0.99,
        ucb_epsilon=0.5,
        ucb_beta=1,
        ucb_window_size=90,

        sync_actor_model_interval=100,

        # other
        processor=None,
        step_interval=1,
        enable_add_episode_end_frame=True,
        test_policy=0,

        # その他
        verbose=1,
    ):

        if not enable_intrinsic_actval_model:
            uvfa_int = []

        self.kwargs = {
            "input_shape": input_shape,
            "input_type": input_type,
            "nb_actions": nb_actions,
            "memory": memory,
            "memory_kwargs": memory_kwargs,
            "actors": actors,

            "input_model": input_model,
            "input_model_emb": input_model_emb,
            "input_model_rnd": input_model_rnd,
            
            "optimizer_ext": optimizer_ext,
            "optimizer_int": optimizer_int,
            "optimizer_emb": optimizer_emb,
            "optimizer_rnd": optimizer_rnd,

            "batch_size": batch_size,
            "input_sequence": input_sequence,
            "dense_units_num": dense_units_num,
            "enable_dueling_network": enable_dueling_network,
            "dueling_network_type": dueling_network_type,
            "lstm_type": lstm_type,
            "lstm_units_num": lstm_units_num,
            "lstmful_input_length": lstmful_input_length,
            
            "memory_warmup_size": memory_warmup_size,
            "target_model_update_interval": target_model_update_interval,
            "enable_double_dqn": enable_double_dqn,
            "enable_rescaling": enable_rescaling,
            "rescaling_epsilon": rescaling_epsilon,
            "priority_exponent": priority_exponent,
            "burnin_length": burnin_length,
            "reward_multisteps": reward_multisteps,
            
            "demo_memory": demo_memory,
            "demo_memory_kwargs": demo_memory_kwargs,
            "demo_episode_dir": demo_episode_dir,
            "demo_ratio_initial": demo_ratio_initial,
            "demo_ratio_final": demo_ratio_final,
            "demo_ratio_steps": demo_ratio_steps,
            
            "episode_memory": episode_memory,
            "episode_memory_kwargs": episode_memory_kwargs,
            "episode_ratio": episode_ratio,
            "episode_verbose": episode_verbose,
            
            "uvfa_ext": uvfa_ext,
            "uvfa_int": uvfa_int,
            "enable_intrinsic_actval_model": enable_intrinsic_actval_model,
            "int_episode_reward_k": int_episode_reward_k,
            "int_episode_reward_epsilon": int_episode_reward_epsilon,
            "int_episode_reward_c": int_episode_reward_c,
            "int_episode_reward_max_similarity": int_episode_reward_max_similarity,
            "int_episode_reward_cluster_distance": int_episode_reward_cluster_distance,
            "int_episodic_memory_capacity": int_episodic_memory_capacity,
            "rnd_err_capacity": rnd_err_capacity,
            "rnd_max_reward": rnd_max_reward,
            "policy_num": policy_num,
            "beta_max": beta_max,
            "gamma0": gamma0,
            "gamma1": gamma1,
            "gamma2": gamma2,
            "ucb_epsilon": ucb_epsilon,
            "ucb_beta": ucb_beta,
            "ucb_window_size": ucb_window_size,

            "step_interval": step_interval,
            "sync_actor_model_interval": sync_actor_model_interval,
            "enable_add_episode_end_frame": enable_add_episode_end_frame,
            "test_policy": test_policy,

            "processor": processor,
            "verbose": verbose,
        }

        self.learner_ps = None
        self.actors_ps = []

    def __del__(self):
        if self.learner_ps is not None:
            self.learner_ps.terminate()
            self.learner_ps = None
        for p in self.actors_ps:
            p.terminate()
        self.actors_ps = []

    def train(self, 
            nb_trains,
            nb_time,
            manager_allocate="/device:CPU:0",
            learner_allocate="/device:GPU:0",
            callbacks=[],
        ):
        
        # GPU確認
        # 参考: https://qiita.com/studio_haneya/items/4dfaf2fb2ac44818e7e0
        if len(tf.config.experimental.list_physical_devices('GPU')) > 0:
            self.enable_GPU = True
            print("GPU: enable")
        else:
            self.enable_GPU = False
            print("GPU: disable")

        #--- init
        self.kwargs["nb_trains"] = nb_trains
        self.kwargs["nb_time"] = nb_time
        self.kwargs["callbacks"] = DisCallbackList(callbacks)
        actor_num = len(self.kwargs["actors"])
        verbose = self.kwargs["verbose"]

        if self.enable_GPU:
            self._train_allocate(manager_allocate, actor_num, learner_allocate, verbose)
        else:
            self._train(actor_num, learner_allocate, verbose)

    def _train_allocate(self, allocate, *args):
        with tf.device(allocate):
            self._train(*args)

    def _train(self, actor_num, learner_allocate, verbose):
    
        # 通信用変数
        self.learner_end_signal = mp.Value(ctypes.c_bool, False)
        self.is_learner_end = mp.Value(ctypes.c_bool, False)
        self.train_count = mp.Value(ctypes.c_int, 0)

        # 経験通信用
        exp_q = mp.Queue()
        
        weights_qs = []
        self.is_actor_ends = []
        for _ in range(actor_num):
            # model weights通信用
            weights_q = mp.Queue()
            weights_qs.append(weights_q)
            self.is_actor_ends.append(mp.Value(ctypes.c_bool, False))

        callbacks = self.kwargs["callbacks"]
        callbacks.on_dis_train_begin()
        t0 = time.time()
        try:

            # learner ps の実行
            learner_args = (
                self.kwargs,
                exp_q,
                weights_qs,
                self.learner_end_signal,
                self.is_learner_end,
                self.train_count,
            )
            if self.enable_GPU:
                learner_args = (learner_allocate,) + learner_args
                self.learner_ps = mp.Process(target=learner_run_allocate, args=learner_args)
            else:
                self.learner_ps = mp.Process(target=learner_run, args=learner_args)
            self.learner_ps.start()

            # actor ps の実行
            self.actors_ps = []
            for i in range(actor_num):
                # args
                actor_args = (
                    i,
                    self.kwargs,
                    exp_q,
                    weights_qs[i],
                    self.is_learner_end,
                    self.train_count,
                    self.is_actor_ends[i],
                )
                if self.enable_GPU:
                    actor = self.kwargs["actors"][i]
                    actor_args = (actor.allocate(i, actor_num),) + actor_args
                    ps = mp.Process(target=actor_run_allocate, args=actor_args)
                else:
                    ps = mp.Process(target=actor_run, args=actor_args)
                self.actors_ps.append(ps)
                ps.start()

            # 終了を待つ
            while True:
                time.sleep(1)  # polling time

                # learner終了確認
                if self.is_learner_end.value:
                    break

                # actor終了確認
                f = True
                for is_actor_end in self.is_actor_ends:
                    if not is_actor_end.value:
                        f = False
                        break
                if f:
                    break
        
        except KeyboardInterrupt:
            pass
        except Exception:
            print(traceback.format_exc())
        if verbose > 0:
            print("done, took {:.3f} seconds".format(time.time() - t0))

        callbacks.on_dis_train_end()
        
        # learner に終了を投げる
        self.learner_end_signal.value = True

        # learner が終了するまで待つ
        t0 = time.time()
        while not self.is_learner_end.value:
            if time.time() - t0 < 60*10:  # timeout
                if verbose > 0:
                    
                    print("learner end timeout.")
                    break
            time.sleep(1)

    def createTestAgent(self, test_actor, learner_model_path):
        return Agent57.createTestAgentStatic(self.kwargs, test_actor, learner_model_path)
        
    @staticmethod
    def createTestAgentStatic(manager_kwargs, test_actor, learner_model_path):
        if not os.path.isfile(learner_model_path):
            return None
        test_actor = ActorRunner(0, manager_kwargs, test_actor(), None, None, None, None, is_test=True)
        test_actor.load_weights(learner_model_path)
        return test_actor



#---------------------------------------------------
# learner
#---------------------------------------------------
def learner_run_allocate(allocate, *args):
    with tf.device(allocate):
        learner_run(*args)

def learner_run(
        kwargs, 
        exp_q,
        weights_qs,
        learner_end_signal,
        is_learner_end,
        train_count,
    ):
    nb_trains = kwargs["nb_trains"]
    nb_time = kwargs["nb_time"]
    verbose = kwargs["verbose"]
    callbacks = kwargs["callbacks"]

    try:
        runner = LearnerRunner(kwargs, exp_q, weights_qs)
        callbacks.on_dis_learner_begin(runner)

        # learner はひたすら学習する
        if verbose > 0:
            print("Learner Start!")
        t0 = time.time()

        while True:
            callbacks.on_dis_learner_train_begin(runner)
            runner.train()
            callbacks.on_dis_learner_train_end(runner)
            train_count.value = runner.learner.train_count

            # 終了判定
            if learner_end_signal.value:
                break

            # 終了判定
            if nb_trains > 0:
                if runner.learner.train_count > nb_trains:
                    break
            if nb_time < time.time() - t0:
                break
            
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
    finally:
        try:
            if verbose > 0:
                print("Learning End. Train Count:{}".format(train_count.value))
            callbacks.on_dis_learner_end(runner)
        except Exception:
            print(traceback.format_exc())
        finally:
            is_learner_end.value = True


class LearnerRunner():
    def __init__(self, kwargs, exp_q, weights_qs):
        self.exp_q = exp_q
        self.weights_qs = weights_qs
        self.kwargs = kwargs

        self.sync_actor_model_interval = kwargs["sync_actor_model_interval"]
        self.enable_intrinsic_actval_model = kwargs["enable_intrinsic_actval_model"]
        if self.enable_intrinsic_actval_model or (UvfaType.REWARD_INT in kwargs["uvfa_ext"]):
            self.enable_intrinsic_reward = True
        else:
            self.enable_intrinsic_reward = False

        self.model_builder = ModelBuilder(
            kwargs["input_shape"],
            kwargs["input_type"],
            kwargs["input_model"],
            kwargs["input_model_emb"],
            kwargs["input_model_rnd"],
            kwargs["batch_size"],
            kwargs["nb_actions"],
            kwargs["input_sequence"],
            kwargs["enable_dueling_network"],
            kwargs["dueling_network_type"],
            kwargs["dense_units_num"],
            kwargs["lstm_type"],
            kwargs["lstm_units_num"],
            kwargs["policy_num"],
        )

        # learner
        self.learner = Learner(
            kwargs["batch_size"],
            kwargs["nb_actions"],
            kwargs["target_model_update_interval"],
            kwargs["enable_double_dqn"],
            kwargs["enable_intrinsic_actval_model"],
            kwargs["lstm_type"],
            kwargs["memory"],
            kwargs["memory_kwargs"],
            kwargs["memory_warmup_size"],
            self.model_builder,
            kwargs["optimizer_ext"],
            kwargs["optimizer_int"],
            kwargs["optimizer_rnd"],
            kwargs["optimizer_emb"],
            kwargs["demo_memory"],
            kwargs["demo_memory_kwargs"],
            kwargs["demo_ratio_initial"],
            kwargs["demo_ratio_steps"],
            kwargs["demo_ratio_final"],
            kwargs["episode_memory"],
            kwargs["episode_memory_kwargs"],
            kwargs["episode_ratio"],
            kwargs["episode_verbose"],
            kwargs["reward_multisteps"],
            kwargs["burnin_length"],
            kwargs["lstmful_input_length"],
            kwargs["priority_exponent"],
            kwargs["input_sequence"],
            kwargs["policy_num"],
            kwargs["beta_max"],
            kwargs["gamma0"],
            kwargs["gamma1"],
            kwargs["gamma2"],
            kwargs["uvfa_ext"],
            kwargs["uvfa_int"],
            len(kwargs["actors"]),
        )

        # demo memory
        if self.learner.demo_memory is not None:
            add_memory(kwargs["demo_episode_dir"], self.learner.demo_memory, self.model_builder, kwargs)
        

    def train(self):
        _train_count = self.learner.train_count + 1
        
        # 一定毎に Actor に weights を送る
        if _train_count % self.sync_actor_model_interval == 0:
            d = {
                "ext": self.learner.actval_ext_model.get_weights()
            }
            if self.enable_intrinsic_actval_model:
                d["int"] = self.learner.actval_int_model.get_weights()
            if self.enable_intrinsic_reward:
                d["rnd_train"]  = self.learner.rnd_train_model.get_weights()
                d["rnd_target"] = self.learner.rnd_target_model.get_weights()
                self.model_builder.sync_embedding_model(self.learner.emb_train_model, self.learner.emb_model)
                d["emb"] = self.learner.emb_model.get_weights()

            for q in self.weights_qs:
                # 空にする
                for _ in range(q.qsize()):
                    q.get(timeout=1)
                # 送る
                q.put(d)
        
        # experience があれば RemoteMemory に追加
        for _ in range(self.exp_q.qsize()):
            exp = self.exp_q.get(timeout=1)
            self.learner.add_exp(exp)
                    
        # train
        self.learner.train()

    def save_weights(self, filepath, overwrite=False, save_memory=False):
        self.learner.save_weights(filepath, overwrite, save_memory)
        
    def load_weights(self, filepath, load_memory=False):
        self.learner.load_weights(filepath, load_memory)


#---------------------------------------------------
# actor
#---------------------------------------------------
class ActorStop(rl.callbacks.Callback):
    def __init__(self, is_learner_end):
        self.is_learner_end = is_learner_end

    def on_step_end(self, episode, logs={}):
        if self.is_learner_end.value:
            raise KeyboardInterrupt()

class ActorUser():
    @staticmethod
    def allocate(actor_index, actor_num):
        return "/device:CPU:0"

    def getPolicy(self, actor_index, actor_num):
        raise NotImplementedError()

    def fit(self, index, agent):
        raise NotImplementedError()


def actor_run_allocate(allocate, *args):
    with tf.device(allocate):
        actor_run(*args)

def actor_run(
        actor_index,
        kwargs, 
        exp_q,
        weights_q,
        is_learner_end,
        train_count,
        is_actor_end,
    ):

    verbose = kwargs["verbose"]
    callbacks = kwargs["callbacks"]

    actor_user = kwargs["actors"][actor_index]()

    runner = ActorRunner(
        actor_index,
        kwargs,
        actor_user,
        exp_q,
        weights_q,
        is_learner_end,
        train_count,
    )

    try:
        callbacks.on_dis_actor_begin(actor_index, runner)

        # run
        if verbose > 0:
            print("Actor{} Start!".format(actor_index))
        actor_user.fit(actor_index, runner)
        
    except KeyboardInterrupt:
        pass
    except Exception:
        print(traceback.format_exc())
        
    try:
        if verbose > 0:
            print("Actor{} End!".format(actor_index))
        callbacks.on_dis_actor_end(actor_index, runner)
    except Exception:
        print(traceback.format_exc())

    is_actor_end.value = True




class ActorRunner(rl.core.Agent):
    def __init__(self, 
            actor_index,
            kwargs,
            actor_user,
            exp_q,
            weights_q,
            is_learner_end,
            train_count,
            is_test=False,
        ):
        super(ActorRunner, self).__init__(kwargs["processor"])
        self.compiled = True  # super
        self.is_test = is_test

        self.is_learner_end = is_learner_end
        self.train_count = train_count
        self.callbacks = kwargs.get("callbacks", [])

        self.actor_index = actor_index
        self.actor_user = actor_user
        self.exp_q = exp_q
        self.weights_q = weights_q
        self.actor_num = len(kwargs["actors"])

        self.model_builder = ModelBuilder(
            kwargs["input_shape"],
            kwargs["input_type"],
            kwargs["input_model"],
            kwargs["input_model_emb"],
            kwargs["input_model_rnd"],
            kwargs["batch_size"],
            kwargs["nb_actions"],
            kwargs["input_sequence"],
            kwargs["enable_dueling_network"],
            kwargs["dueling_network_type"],
            kwargs["dense_units_num"],
            kwargs["lstm_type"],
            kwargs["lstm_units_num"],
            kwargs["policy_num"],
        )
        
        self.actor = Actor(
            kwargs["input_shape"],
            kwargs["input_sequence"],
            kwargs["nb_actions"],
            actor_user.getPolicy(actor_index, self.actor_num),
            kwargs["batch_size"],
            kwargs["lstm_type"],
            kwargs["reward_multisteps"],
            kwargs["lstmful_input_length"],
            kwargs["burnin_length"],
            kwargs["enable_intrinsic_actval_model"],
            kwargs["enable_rescaling"],
            kwargs["priority_exponent"],
            kwargs["int_episode_reward_k"],
            kwargs["int_episode_reward_epsilon"],
            kwargs["int_episode_reward_c"],
            kwargs["int_episode_reward_max_similarity"],
            kwargs["int_episode_reward_cluster_distance"],
            kwargs["int_episodic_memory_capacity"],
            kwargs["rnd_err_capacity"],
            kwargs["rnd_max_reward"],
            kwargs["policy_num"],
            kwargs["test_policy"],
            kwargs["beta_max"],
            kwargs["gamma0"],
            kwargs["gamma1"],
            kwargs["gamma2"],
            kwargs["ucb_epsilon"],
            kwargs["ucb_beta"],
            kwargs["ucb_window_size"],
            self.model_builder,
            kwargs["uvfa_ext"],
            kwargs["uvfa_int"],
            actor_index,
        )
        
        # local
        self.enable_intrinsic_actval_model = kwargs["enable_intrinsic_actval_model"]
        if self.enable_intrinsic_actval_model or (UvfaType.REWARD_INT in kwargs["uvfa_ext"]):
            self.enable_intrinsic_reward = True
        else:
            self.enable_intrinsic_reward = False
        self.step_interval = kwargs["step_interval"]
        self.enable_add_episode_end_frame = kwargs["enable_add_episode_end_frame"]

        # model share
        self.actor.build_model(None)
        

    def reset_states(self):  # override
        self.actor.training = self.training  # training はコンストラクタで初期化されない
        self.actor.episode_begin()

        self.local_step = 0
        self.repeated_action = 0
        self.step_reward = 0
        self.recent_terminal = False
    
    def compile(self, optimizer, metrics=[]):  # override
        self.compiled = True  # super

    def save_weights(self, filepath, overwrite=False):  # override
        pass

    def load_weights(self, filepath):  # override
        self.actor.load_weights(filepath)

    def forward(self, observation):  # override
        
        # フレームスキップ(action_interval毎に行動を選択する)
        action = self.repeated_action
        if self.recent_terminal or (self.local_step % self.step_interval == 0):
            self.actor.forward_train_before(observation)

            if self.recent_terminal and self.enable_add_episode_end_frame:
                # 最終フレーム後に1フレーム追加
                exp = self.actor.create_exp(True, update_terminal=False)
                if exp is not None:
                    self.exp_q.put(exp)
                    exp = None
                self.actor.add_episode_end_frame()
                exp = self.actor.create_exp(True, update_terminal=True)
                if exp is not None:
                    self.exp_q.put(exp)
                    exp = None
            else:
                exp = self.actor.create_exp(True)
                if exp is not None:
                    self.exp_q.put(exp)
                    exp = None

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
        if not self.training:
            return []

        # weightが届いていればmodelを更新
        if not self.weights_q.empty():
            d = self.weights_q.get(timeout=1)
            self.actor.actval_ext_model.set_weights(d["ext"])
            if self.enable_intrinsic_actval_model:
                self.actor.actval_int_model.set_weights(d["int"])
            if self.enable_intrinsic_reward:
                self.actor.rnd_train_model.set_weights(d["rnd_train"])
                self.actor.rnd_target_model.set_weights(d["rnd_target"])
                self.actor.emb_model.set_weights(d["emb"])
            d = None
        
        return []

    @property
    def layers(self):  # override
        return []


    def fit(self, env, nb_steps=99_999_999_999, callbacks=[], **kwargs):  # override
        try:
            if self.is_test:
                super().fit(nb_steps, callbacks, **kwargs)
                return

            callbacks.extend(self.callbacks.callbacks)

            # stop
            callbacks.append(ActorStop(self.is_learner_end))

            # keras-rlでの学習
            super().fit(env, nb_steps=nb_steps, callbacks=callbacks, **kwargs)

        except Exception:
            print(traceback.print_exc())

       


# distributing callback
class DisCallback(rl.callbacks.Callback):
    def __init__(self):
        pass

    def on_dis_train_begin(self):
        pass

    def on_dis_train_end(self):
        pass

    def on_dis_learner_begin(self, learner):
        pass
    
    def on_dis_learner_end(self, learner):
        pass

    def on_dis_learner_train_begin(self, learner):
        pass

    def on_dis_learner_train_end(self, learner):
        pass

    def on_dis_actor_begin(self, actor_index, runner):
        pass

    def on_dis_actor_end(self, actor_index, runner):
        pass

class DisCallbackList(DisCallback):
    def __init__(self, callbacks):
        self.callbacks = callbacks

    def on_dis_train_begin(self):
        for callback in self.callbacks:
            callback.on_dis_train_begin()

    def on_dis_train_end(self):
        for callback in self.callbacks:
            callback.on_dis_train_end()

    def on_dis_learner_begin(self, learner):
        for callback in self.callbacks:
            callback.on_dis_learner_begin(learner)
    
    def on_dis_learner_end(self, learner):
        for callback in self.callbacks:
            callback.on_dis_learner_end(learner)

    def on_dis_learner_train_begin(self, learner):
        for callback in self.callbacks:
            callback.on_dis_learner_train_begin(learner)

    def on_dis_learner_train_end(self, learner):
        for callback in self.callbacks:
            callback.on_dis_learner_train_end(learner)

    def on_dis_actor_begin(self, actor_index, runner):
        for callback in self.callbacks:
            callback.on_dis_actor_begin(actor_index, runner)

    def on_dis_actor_end(self, actor_index, runner):
        for callback in self.callbacks:
            callback.on_dis_actor_end(actor_index, runner)

