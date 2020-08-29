import gym
from keras.optimizers import Adam

import traceback
import os

from .dqn import DQN
from .agent57 import Agent57
from .model import InputType, DQNImageModel, LstmType
from .policy import AnnealingEpsilonGreedy
from .memory import PERRankBaseMemory, PERProportionalMemory
from .env_play import EpisodeSave, EpisodeReplay

from .callbacks import ConvLayerView, MovieLogger
from .callbacks import LoggerType, TimeStop, TrainLogger, ModelIntervalCheckpoint
from .callbacks import DisTrainLogger, DisSaveManager




def run_gym_dqn(
        enable_train,
        env,
        env_name,
        kwargs, 
        nb_steps=999_999_999,
        nb_time=999_999_999,
        logger_type=LoggerType.STEP,
        log_interval=0,
        test_env=None,
        test_episodes=10,
        load_weights="",
        checkpoint_interval=0,
        movie_save=False,
        base_dir="tmp",
    ):

    os.makedirs(base_dir, exist_ok=True)
    weight_file = os.path.join(base_dir, "{}_weight.h5".format(env_name))
    print("nb_steps: {}".format(nb_steps))
    print("nb_time : {:.2f}m".format(nb_time/60))
    
    agent = DQN(**kwargs)
    print(agent.actor.actval_ext_model.summary())

    if test_env is None:
        test_agent = None
    else:
        test_agent = DQN(**kwargs)
    log = TrainLogger(
        logger_type,
        interval=log_interval,
        savefile=os.path.join(base_dir, "{}_log.json".format(env_name)),
        test_agent=test_agent,
        test_env=test_env,
        test_episodes=test_episodes,
        test_save_max_reward_file=weight_file+'_max_{step:02d}_{reward}.h5'
    )

    if enable_train:
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        try:
            callbacks = [log]
            if load_weights != "":
                agent.load_weights(load_weights, load_memory=True)

            if checkpoint_interval > 0:
                callbacks.append(
                    ModelIntervalCheckpoint(
                        filepath = weight_file + '_{step:02d}.h5',
                        interval=checkpoint_interval,
                        save_memory=False,
                    )
                )

            callbacks.append(TimeStop(nb_time))

            agent.fit(env, nb_steps=nb_steps, visualize=False, verbose=0, callbacks=callbacks)

        except Exception:
            print(traceback.print_exc())
            raise

        # save
        print("weight save: " + weight_file)
        agent.save_weights(weight_file, overwrite=True, save_memory=False)
        
    # plt
    log.drawGraph("step")
    
    # 訓練結果を見る
    if log.max_reward_file == "":
        print("weight load: " + weight_file)
        agent.load_weights(weight_file)
    else:
        print("weight load: " + log.max_reward_file)
        agent.load_weights(log.max_reward_file)
    agent.test(env, nb_episodes=5, visualize=True)

    # 動画保存用
    if movie_save:
        movie = MovieLogger()
        callbacks = [movie]
        if kwargs["input_type"] != InputType.VALUES:
            conv = ConvLayerView(agent)
            callbacks.append(conv)
        agent.test(env, nb_episodes=1, visualize=False, callbacks=callbacks)
        movie.save(gifname="tmp/{}_1.gif".format(env_name), fps=30)
        if kwargs["input_type"] != InputType.VALUES:
            conv.save(grad_cam_layers=["c3"], add_adv_layer=True, add_val_layer=True, 
                end_frame=200, gifname="tmp/{}_2.gif".format(env_name), fps=10)

    env.close()



def run_gym_agent57(
        enable_train,
        env,
        env_name,
        kwargs,
        nb_trains=999_999_999,
        nb_time=999_999_999,
        logger_type=LoggerType.TIME,
        log_interval=0,
        test_env=None,
        test_episodes=10,
        is_load_weights=False,
        checkpoint_interval=0,
        movie_save=False,
    ):
    base_dir = os.path.join("tmp_{}".format(env_name))
    os.makedirs(base_dir, exist_ok=True)
    print("nb_time  : {:.2f}m".format(nb_time/60))
    print("nb_trains: {}".format(nb_trains))
    weight_file = os.path.join(base_dir, "{}_weight.h5".format(env_name))

    manager = Agent57(**kwargs)

    if test_env is None:
        test_actor = None
    else:
        test_actor = kwargs["actors"][0]
    log = DisTrainLogger(
        logger_type,
        interval=log_interval,
        savedir=base_dir,
        test_actor=test_actor,
        test_env=test_env,
        test_episodes=test_episodes,
        test_save_max_reward_file=os.path.join(base_dir, 'max_{step:02d}_{reward}.h5')
    )

    if enable_train:
        print("--- start ---")
        print("'Ctrl + C' is stop.")
        save_manager = DisSaveManager(
            save_dirpath=base_dir,
            is_load=is_load_weights,
            save_memory=False,
            checkpoint=(checkpoint_interval>0),
            checkpoint_interval=checkpoint_interval,
            verbose=0
        )

        manager.train(nb_trains, nb_time, callbacks=[save_manager, log])

    # plt
    log.drawGraph("train")

    # 訓練結果を見る
    agent = manager.createTestAgent(kwargs["actors"][0], "tmp_{}/last/learner.dat".format(env_name))
    if agent is None:
        return
    agent.test(env, nb_episodes=5, visualize=True)

    # 動画保存用
    if movie_save:
        movie = MovieLogger()
        callbacks = [movie]
        if kwargs["input_type"] != InputType.VALUES:
            conv = ConvLayerView(agent)
            callbacks.append(conv)
        agent.test(env, nb_episodes=1, visualize=False, callbacks=callbacks)
        movie.save(gifname="tmp/{}_1.gif".format(env_name), fps=30)
        if kwargs["input_type"] != InputType.VALUES:
            conv.save(grad_cam_layers=[], add_adv_layer=True, add_val_layer=True, 
                end_frame=200, gifname="tmp/{}_2.gif".format(env_name), fps=10)

    env.close()


def run_play(env, episode_save_dir, processor, **kwargs):
    es = EpisodeSave(env, 
        episode_save_dir=episode_save_dir,
        processor=processor,
        font="arial")
    es.play(**kwargs)
    env.close()


def run_replay(episode_save_dir, **kwargs):
    r = EpisodeReplay(episode_save_dir, font="arial")
    r.play(**kwargs)




