import gym
import pygame
import numpy as np
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.animation

import os
import pickle
from tkinter import Tk
from tkinter import messagebox
import glob
import time

from .model import LstmType
from .model import ModelBuilder
from .actor import Actor
from .policy import EpsilonGreedy

class _PlayWindow():
    """ copy from: https://github.com/openai/gym/blob/master/gym/utils/play.py """

    def __init__(self, font=None):
        self.font = font

        self.min_height = 600
        self.info_width = 200

    def play(self, fps=30, zoom=None, keys_to_action=None):
        pygame.init()
        clock = pygame.time.Clock()

        self.fps = fps
        self.msgs = {
            "key": [
                "Available keys:",
                "ESC: exit",
                "1-6: size change"
            ],
        }
        font1 = pygame.font.SysFont(self.font, 22)

        self.org_size = self.on_play_before()
        self.screen = pygame.display.set_mode(self.org_size)
        self.resize(zoom)

        self.running = True

        while self.running:
            self.screen.fill((0, 0, 0, 0))

            # process pygame events
            for event in pygame.event.get():
                self.on_event_loop(event)
            
            self.on_loop()

            # draw text
            y = 5
            for v in self.msgs.values():
                for s in v:
                    self.screen.blit(font1.render(s, False, (255,255,255)), (5+ self.video_size[0], y))
                    y += 25
                y += 10
            
            pygame.display.flip()
            clock.tick(self.fps)
        pygame.quit()
        self.on_play_end()

    def on_play_before(self):
        raise NotImplementedError()

    def on_play_end(self):
        pass

    def on_event_loop(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == 27:  # ESC
                self.running = False
            elif event.unicode == '1':
                self.resize(0.5)
            elif event.unicode == '2':
                self.resize(1.0)
            elif event.unicode == '3':
                self.resize(1.5)
            elif event.unicode == '4':
                self.resize(2.0)
            elif event.unicode == '5':
                self.resize(3.0)
            elif event.unicode == '6':
                self.resize(4.0)
        elif event.type == pygame.QUIT:
            self.running = False

    def on_loop(self):
        pass
    
    def resize(self, zoom):
        if zoom is None:
            zoom = 1
        h = self.org_size[1] * zoom
        self.video_size = int(self.org_size[0] * zoom), int(h)
        if h < self.min_height:
            h = self.min_height
        window_size = int(self.video_size[0] + self.info_width), int(h)
        pygame.display.set_mode(window_size)
        self.set_msg(["size: {}".format(self.video_size)])

    def display_arr(self, arr):
        arr_min, arr_max = arr.min(), arr.max()
        arr = 255.0 * (arr - arr_min) / (arr_max - arr_min)
        pyg_img = pygame.surfarray.make_surface(arr.swapaxes(0, 1))
        pyg_img = pygame.transform.scale(pyg_img, self.video_size)
        self.screen.blit(pyg_img, (0,0))

    def add_key_msg(self, msg):
        self.msgs["key"].append(msg)

    def set_msg(self, msgs):
        print("\n".join(msgs))
        self.msgs["msg"] = msgs

    def set_info(self, msgs):
        self.msgs["info"] = msgs



class EpisodeSave(_PlayWindow):
    def __init__(self, env, processor=None, episode_save_dir=None, keys_to_action=None, **kwargs):
        super().__init__(**kwargs)
        self.env = env
        self.processor = processor
        self.episode_save_dir = episode_save_dir
        self.keys_to_action = keys_to_action
        if self.episode_save_dir is not None:
            os.makedirs(self.episode_save_dir, exist_ok=True)


    def on_play_before(self):
        self.env.reset()
        rendered = self.env.render( mode='rgb_array')
        env_size = [rendered.shape[1], rendered.shape[0]]
        
        if self.keys_to_action is None:
            if self.processor is not None and hasattr(self.processor, 'get_keys_to_action'):
                self.keys_to_action = self.processor.get_keys_to_action()
            elif hasattr(self.env, 'get_keys_to_action'):
                self.keys_to_action = self.env.get_keys_to_action()
            elif hasattr(self.env.unwrapped, 'get_keys_to_action'):
                self.keys_to_action = self.env.unwrapped.get_keys_to_action()
            else:
                assert False, "env does not have explicit key to action mapping, " + \
                            "please specify one manually"
        self.relevant_keys = set(sum(map(list, self.keys_to_action.keys()),[]))
        super().add_key_msg("-/+: speed change")
        super().add_key_msg("f: frameadvance")
        super().add_key_msg("p: Pause/Unpouse")
        super().add_key_msg("Use keys:")
        for k in self.relevant_keys:
            super().add_key_msg(" {}".format(pygame.key.name(k)))

        self.pressed_keys = []
        self.env_done = True
        self.episode_count = 0
        self.is_frameadvance = False

        return env_size

    

    def on_event_loop(self, event):
        super().on_event_loop(event)

        if event.type == pygame.KEYDOWN:
            if event.key in self.relevant_keys:
                self.pressed_keys.append(event.key)
            elif event.unicode == 'p':
                self.env_pause = False if self.env_pause else True
            elif event.unicode == 'f':
                self.is_frameadvance = True
            elif event.unicode == '+':
                self.fps += 10
                super().set_msg(["fps: {}".format(self.fps)])
            elif event.unicode == '-':
                self.fps -= 10
                if self.fps < 10:
                    self.fps = 10
                super().set_msg(["fps: {}".format(self.fps)])
        elif event.type == pygame.KEYUP:
            if event.key in self.relevant_keys:
                self.pressed_keys.remove(event.key)

    def on_loop(self):
        super().on_loop()

        if self.env_done:
            self.env_done = False
            self.env_pause = True
            self.obs = self.env.reset()
            self.episode_count += 1
            self.total_reward = 0
            self.step = 0
            self.action = 0
            self.reward = 0
            self.states1 = []
            self.states2 = []
            super().set_msg(["start new episode {}.".format(self.episode_count)])
        
        f = True
        if self.env_pause:
            f = False
        if self.is_frameadvance:
            f = True
            self.env_pause = True
            self.is_frameadvance = False
        
        if f:
            self.action = self.keys_to_action.get(tuple(sorted(self.pressed_keys)), 0)
            if self.processor is not None:
                _action = self.processor.process_action(self.action)
            else:
                _action = self.action
            self.obs, self.reward, self.env_done, info = self.env.step(_action)
            if self.processor is not None:
                self.obs, self.reward, self.env_done, info = \
                    self.processor.process_step(self.obs, self.reward, self.env_done, info)
            self.total_reward += self.reward
            self.step += 1

            self.states1.append({
                "action": self.action,
                "observation": self.obs,
                "reward": self.reward,
                "done": self.env_done,
            })
            self.states2.append({
                "step": self.step,
                "reward_total": self.total_reward,
                "info": info,
                "rgb": self.env.render(mode='rgb_array'),
            })

            if self.env_done:
                Tk().wm_withdraw() #to hide the main window
                if messagebox.askyesno("Save?", "Do you want to save the episode?"):
                    path1 = os.path.join(self.episode_save_dir, "episode{}.dat".format(self.episode_count))
                    path2 = os.path.join(self.episode_save_dir, "episode{}.dat.display".format(self.episode_count))
                    super().set_msg(["save: {}".format(path1)])
                    with open(path1, 'wb') as f:
                        pickle.dump(self.states1, f)

                    d = {
                        "episode": self.episode_count,
                        "rgb_size": self.org_size,
                        "states": self.states2,
                    }
                    with open(path2, 'wb') as f:
                        pickle.dump(d, f)
                else:
                    self.episode_count -= 1

        if self.obs is not None:
            rendered = self.env.render(mode='rgb_array')
            super().display_arr(rendered)

        # env info
        super().set_info([
            "episode: {}".format(self.episode_count),
            "action : {}".format(self.action),
            "step   : {}".format(self.step),
            "reward : {}".format(self.reward),
            "total  : {}".format(self.total_reward),
        ])



class EpisodeReplay(_PlayWindow):
    def __init__(self, episode_save_dir=None, **kwargs):
        super().__init__(**kwargs)
        self.episode_save_dir = episode_save_dir

        path = os.path.join(self.episode_save_dir, "episode1.dat")
        assert os.path.isfile(path), "episode is not found: {}".format(path)


    def on_play_before(self):

        super().add_key_msg("down: prev episode")
        super().add_key_msg("up: next episode")
        super().add_key_msg("left: prev frame")
        super().add_key_msg("right: next frame")
        super().add_key_msg("p: Pause/Unpouse")

        self.episode = 1
        self.set_episode()

        return self.org_size
    
    def on_event_loop(self, event):
        super().on_event_loop(event)
        
        if event.type == pygame.KEYDOWN:
            if event.unicode == 'p':
                self.env_pause = False if self.env_pause else True
            elif event.key == 275:  # right
                self.step += 1
            elif event.key == 276:  # left
                self.step -= 1
            elif event.key == 273:  # up
                self.episode += 1
                self.set_episode()
            elif event.key == 274:  # down
                self.episode -= 1
                self.set_episode()
        

    def on_loop(self):
        super().on_loop()

        if self.step < 0:
            self.step = 0
            self.env_pause = True
        if self.step >= len(self.states1):
            self.step = len(self.states1) - 1
            self.env_pause = True
        state1 = self.states1[self.step]
        state2 = self.states2[self.step]
        
        super().display_arr(state2["rgb"])
        super().set_info([
            "episode: {}".format(self.episode),
            "action : {}".format(state1["action"]),
            "step   : {} / {}".format(state2["step"], len(self.states1)),
            "reward : {}".format(state1["reward"]),
            "total  : {}".format(state2["reward_total"]),
            "done   : {}".format(state1["done"]),
        ])
        
        if not self.env_pause:
            self.step += 1
    
    def set_episode(self):
        if self.episode < 1:
            self.episode = 1

        episode_file = os.path.join(self.episode_save_dir, "episode{}.dat".format(self.episode))
        path1 = episode_file
        path2 = episode_file + ".display"
        if not os.path.isfile(path1):
            super().set_msg(["episode file is not found: {}".format(path1)])
            return
        with open(path1, 'rb') as f:
            self.states1 = pickle.load(f)
        if os.path.isfile(path2):
            with open(path2, 'rb') as f:
                d = pickle.load(f)
            self.org_size = d["rgb_size"]
            self.states2 = d["states"]
        else:
            print("display file is not found: {}".format(path2))

        self.env_pause = True
        self.step = 0
        super().set_msg(["episode{} is load.".format(self.episode)])
    

    def save(self,
            episode,
            start_frame=0,
            end_frame=0,
            gifname="",
            mp4name="",
            interval=200,
            fps=30
        ):
        #--- episode を読み込む
        self.episode = episode
        self.set_episode()
        self.frames = []
        for s in self.states2:
            self.frames.append(s["rgb"])
        
        assert start_frame<len(self.frames), "start frame is over frames({})".format(len(self.frames))
        if end_frame == 0:
          end_frame = len(self.frames)
        elif end_frame > len(self.frames):
            end_frame = len(self.frames)
        self.start_frame = start_frame
        self.t0 = time.time()
        
        self.patch = plt.imshow(self.frames[0])
        plt.axis('off')
        ani = matplotlib.animation.FuncAnimation(plt.gcf(), self._plot, frames=end_frame - start_frame, interval=interval)

        if gifname != "":
            ani.save(gifname, writer="pillow", fps=fps)
            #ani.save(gifname, writer="imagemagick", fps=fps)
        if mp4name != "":
            ani.save(mp4name, writer="ffmpeg")
    
    def _plot(self, frame):
        if frame % 50 == 0:
            print("{}f {:.2f}m".format(frame, (time.time()-self.t0)/60))
        
        #plt.imshow(self.frames[frame + self.start_frame])
        self.patch.set_data(self.frames[frame + self.start_frame])


def add_memory(episode_save_dir, memory, model_builder, kwargs):
    actor = Actor(
        kwargs["input_shape"],
        kwargs["input_sequence"],
        kwargs["nb_actions"],
        EpsilonGreedy(0),
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
        model_builder,
        kwargs["uvfa_ext"],
        kwargs["uvfa_int"],
        0,
    )
    actor.build_model(None)
    step_interval = kwargs["step_interval"]
    input_shape = kwargs["input_shape"]

    for fn in glob.glob(os.path.join(episode_save_dir, "episode*.dat")):
        print("load: {}".format(fn))
        with open(fn, 'rb') as f:
            epi_states = pickle.load(f)
        if len(epi_states) <= 0:
            continue
        
        if input_shape != np.asarray(epi_states[0]["observation"]).shape:
            print("episode shape is not match. input_shape{} != epi_shape{}".format(
                input_shape,
                epi_states[0]["observation"].shape
            ))
            continue

        # init
        actor.training = True
        actor.episode_begin()
        
        # episode
        total_reward = 0
        for step, epi_state in enumerate(epi_states):

            if step % step_interval == 0:
                actor.forward_train_before(epi_state["observation"])
                exp = actor.create_exp(False)
                memory.add(exp)
                actor.forward_train_after()

                # アクションを入れかえる
                actor.recent_actions[-1] = epi_state["action"]

                actor.backward(epi_state["reward"], epi_state["done"])

            # 最後の状態も追加
            if epi_state["done"]:
                actor.forward_train_before(epi_state["observation"])
                exp = actor.create_exp(False)
                memory.add(exp)

            # 表示用
            total_reward += epi_state["reward"]

        print("demo replay loaded, on_memory: {}, total reward: {}".format(len(memory), total_reward))

