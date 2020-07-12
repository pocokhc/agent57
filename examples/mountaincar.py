import gym
from keras.optimizers import Adam

import traceback

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agent.agent57 import ActorUser
from agent.policy import EpsilonGreedy, AnnealingEpsilonGreedy
from agent.memory import PERRankBaseMemory, PERProportionalMemory
from agent.model import InputType, LstmType, DQNImageModel
from agent.common import seed_everything
from agent.callbacks import LoggerType

from agent.main_runner import run_gym_dqn, run_play, run_replay, run_gym_agent57


seed_everything(42)
ENV_NAME = "MountainCar-v0"
episode_save_dir = "tmp_{}.".format(ENV_NAME)


def create_parameter(env):

    warmup = 1000

    kwargs = {
        "input_shape": env.observation_space.shape, 
        "input_type": InputType.VALUES,
        "nb_actions": env.action_space.n,

        "memory": "PERRankBaseMemory",
        "memory_kwargs": {
            "capacity": 60_000,
            "alpha": 1.0,          # PERの確率反映率
            "beta_initial": 0.0,   # IS反映率の初期値(1.0が最大)
            "beta_steps": warmup+50_000,  # IS反映率の上昇step数
            "enable_is": True,     # ISを有効にするかどうか
        },

        "optimizer_ext": Adam(lr=0.001),
        "optimizer_int": Adam(lr=0.001),
        "optimizer_rnd": Adam(lr=0.0001, epsilon=0.0001),
        "optimizer_emb": Adam(lr=0.001, epsilon=0.001),

        # NN
        "batch_size": 16,     # batch_size
        "input_sequence": 4,         # 入力フレーム数
        "dense_units_num": 16,       # dense層のユニット数
        "lstm_type": LstmType.STATELESS,           # 使用するLSTMアルゴリズム
        "lstm_units_num": 16,             # LSTMのユニット数
        "lstmful_input_length": 2,       # ステートフルLSTMの入力数

        # train
        "memory_warmup_size": warmup,    # 初期のメモリー確保用step数(学習しない)
        "target_model_update_interval": 1000,  # target networkのupdate間隔
        "enable_rescaling": True,   # rescalingを有効にするか
        "priority_exponent": 0.9,   # priority優先度
        "burnin_length": 0,        # burn-in期間
        "reward_multisteps": 3,    # multistep reward

        "demo_memory": "PERProportionalMemory",
        "demo_memory_kwargs": {
            "capacity": 100_000,
            "alpha": 0.8,
        },
        "demo_episode_dir": episode_save_dir,
        "demo_ratio_initial": 1.0,
        "demo_ratio_final": 1.0/512.0,
        "demo_ratio_steps": warmup+50_000,

        "episode_memory": "PERProportionalMemory",
        "episode_memory_kwargs": {
            "capacity": 2000,
            "alpha": 0.8,
        },
        "episode_ratio": 1.0/32.0,

        # intrinsic_reward
        "enable_intrinsic_reward": True,
        "policy_num": 8,
        "beta_max": 0.3,
        "ucb_epsilon": 0.2,
        "ucb_window_size": 40,

        "step_interval": 1,
    }

    return kwargs


#---------------------------------------------------------

def run_dqn(enable_train):
    env = gym.make(ENV_NAME)

    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("reward_range      : " + str(env.reward_range))

    kwargs = create_parameter(env)
    kwargs["action_policy"] = AnnealingEpsilonGreedy(
        initial_epsilon=0.5,      # 初期ε
        final_epsilon=0.1,       # 最終状態でのε
        exploration_steps=kwargs["memory_warmup_size"] + 20_000  # 初期→最終状態になるまでのステップ数
    )

    run_gym_dqn(
        enable_train,
        env,
        ENV_NAME,
        kwargs,
        nb_steps=kwargs["memory_warmup_size"] + 50_000,
        nb_time=60*60,
        logger_type=LoggerType.STEP,
        log_interval=1000,
        test_env=env,
        movie_save=False,
    )
    
#---------------------------------------------------------

class MyActor(ActorUser):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)

    def fit(self, index, agent):
        env = gym.make(ENV_NAME)
        agent.fit(env, visualize=False, verbose=0)
        env.close()

class MyActor1(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.01)

class MyActor2(MyActor):
    def getPolicy(self, actor_index, actor_num):
        return EpsilonGreedy(0.1)


def create_env():
    return gym.make(ENV_NAME)


def run_agent57(enable_train):
    env = gym.make(ENV_NAME)

    # ゲーム情報
    print("action_space      : " + str(env.action_space))
    print("observation_space : " + str(env.observation_space))
    print("reward_range      : " + str(env.reward_range))

    kwargs = create_parameter(env)

    kwargs["actors"] = [MyActor1]
    kwargs["sync_actor_model_interval"] = 50  # learner から model を同期する間隔

    run_gym_agent57(
        enable_train,
        env,
        ENV_NAME,
        kwargs,
        nb_trains=30000,
        nb_time=60*60,
        logger_type=LoggerType.STEP,
        log_interval=1000,
        test_env=create_env,
        is_load_weights=False,
        movie_save=False,
    )


#----------------------


if __name__ == '__main__':
    
    # エピソードを作成、保存
    if False:
        env = gym.make(ENV_NAME)
        kwargs = create_parameter(env)
        run_play(env, episode_save_dir, kwargs["processor"])

    # エピソードを再生(確認用)
    if False:
        run_replay(episode_save_dir)

    # SingleActorレーニング
    if False:
        run_dqn(enable_train=True)
        #run_dqn(enable_train=False)  # test only

    # 複数Actorレーニング
    if True:
        run_agent57(enable_train=True)
        #run_agent57(enable_train=False)  # test only

