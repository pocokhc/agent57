import rl
import keras
from keras import backend as K
import matplotlib.pyplot as plt
import matplotlib.animation
import cv2
import numpy as np

import os
import json
import time
import enum
import tempfile
import glob

from .model import LstmType
from .agent57 import Agent57, DisCallback


class MovieLogger(rl.callbacks.Callback):
    def __init__(self, verbose=1):
        self.frames = []
        self.verbose = verbose

    def on_action_end(self, action, logs):  # override
        self.frames.append(self.env.render(mode='rgb_array'))

    def save(self, 
            start_frame=0,
            end_frame=0,
            gifname="",
            mp4name="",
            interval=200,
            fps=30
        ):
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

        return ani
    
    def _plot(self, frame):
        if self.verbose > 0:
            if frame % 50 == 0:
                print("{}f {:.2f}m".format(frame, (time.time()-self.t0)/60))
        
        #plt.imshow(self.frames[frame + self.start_frame])
        self.patch.set_data(self.frames[frame + self.start_frame])


class ConvLayerView(rl.callbacks.Callback):
    def __init__(self, agent):
        self.observations = []
        self.agent = agent

    def on_step_end(self, step, logs):  # override
        self.observations.append(logs["observation"])

    def save(self, grad_cam_layers, add_adv_layer=False, add_val_layer=False, start_frame=0, end_frame=0, gifname="", mp4name="", interval=200, fps=30):
        assert start_frame<len(self.observations)
        if end_frame == 0:
          end_frame = len(self.observations)
        elif end_frame > len(self.observations):
            end_frame = len(self.observations)
        self.start_frame = start_frame
        
        for layer in grad_cam_layers:
            f = False
            for layer2 in self.agent.learner.actval_ext_model.layers:
                if layer == layer2.name:
                    f = True
                    break
            assert f, "layer({}) is not found.".format(layer)

        self.grad_cam_layers = grad_cam_layers
        self.add_adv_layer = add_adv_layer
        self.add_val_layer = add_val_layer
        self._init()

        plt.figure(figsize=(8.0, 6.0), dpi = 100)  # 大きさを指定
        plt.axis('off')
        ani = matplotlib.animation.FuncAnimation(plt.gcf(), self._plot, frames=end_frame - start_frame, interval=interval)

        if gifname != "":
            ani.save(gifname, writer="pillow", fps=fps)
            #ani.save(gifname, writer="imagemagick", fps=fps)
        if mp4name != "":
            ani.save(mp4name, writer="ffmpeg")

        return ani
        #return ani.to_jshtml()

    def _init(self):
        model = self.agent.learner.actval_ext_model

        # 出力毎に関数を用意
        self.grads_funcs = []
        for nb_action in range(self.agent.learner.nb_actions):
            # 各勾配を定義
            class_output = model.output[0][nb_action]

            #--- layes毎に定義
            # Grad_CAM 勾配はlayer->出力層
            outputs = []
            for layer in self.grad_cam_layers:
                output = model.get_layer(layer).output
                outputs.append(output)
                grad = K.gradients(class_output, output)[0]
                outputs.append(grad)

            # SaliencyMap 勾配は入力層->layer
            if self.add_adv_layer:
                # adv層は出力と同じ(action数)なので予測結果を指定
                output = model.get_layer("adv").output[0][nb_action]
                grad = K.gradients(output, model.input)[0]
                outputs.append(grad)
            if self.add_val_layer:
                # v層はUnit数が1つしかないので0を指定
                output = model.get_layer("v").output[0][0]
                grad = K.gradients(output, model.input)[0]
                outputs.append(grad)

            # functionを定義
            grads_func = K.function([model.input, K.learning_phase()],outputs)
            self.grads_funcs.append(grads_func)
        


    def _plot(self, frame):
        if frame % 30 == 0:  # debug
            print(frame)

        observations = self.observations
        input_sequence = self.agent.learner.input_sequence
        model = self.agent.learner.actval_ext_model
        batch_size = self.agent.learner.batch_size

        # 入力分の frame がたまるまで待つ
        if frame + self.start_frame < input_sequence:
            return

        # 予測結果を出す
        if self.agent.learner.lstm_type != LstmType.STATEFUL:
            input_state = np.asarray([observations[frame - input_sequence:frame + self.start_frame]])
            prediction = model.predict(input_state, batch_size=1)[0]
        else:
            input_state = np.asarray(observations[frame - input_sequence:frame + self.start_frame])
            # batchサイズ分増やす
            input_state = np.full((batch_size,)+input_state.shape, input_state)
            prediction = model.predict(input_state, batch_size=batch_size)[0]
        class_idx = np.argmax(prediction)

        #--- オリジナル画像
        # 形式は(w,h)でかつ0～1で正規化されているので画像形式に変換
        org_img = np.asarray(observations[frame])  # (w,h)
        #org_img = np.transpose(org_img, (1, 0))  # (w,h) -> (h,w)
        org_shape = (org_img.shape[1], org_img.shape[0])
        org_img *= 255
        org_img = cv2.cvtColor(np.uint8(org_img), cv2.COLOR_GRAY2BGR)  # (w,h) -> (w,h,3)
        #cv2.imshow('image', org_img)
        #cv2.waitKey(0)
        imgs = [org_img]
        names = ["original"]

        #--- 勾配を計算
        grads = self.grads_funcs[class_idx]([input_state, 0])

        # Grad-CAM
        for i in range(len(self.grad_cam_layers)):
            name = self.grad_cam_layers[i]
            c_output = grads[i*2]
            c_val = grads[i*2+1]
            #c_val = np.transpose(c_val, (1, 0))  # (w,h) -> (h,w)

            cam = self._grad_cam(c_output, c_val, org_img, org_shape)
            imgs.append(cam)
            names.append(name)
        
        # SaliencyMap
        i = len(self.grad_cam_layers)*2
        if self.add_adv_layer:
            c_val = grads[i][0][input_sequence-1]  # input_sequenceあるので最後を取得
            #c_val = np.transpose(c_val, (1, 0))  # (w,h) -> (h,w)

            img = self._saliency_map(c_val, org_img, org_shape)
            imgs.append(img)
            names.append("advance")
            i += 1
        if self.add_val_layer:
            v_val = grads[i][0][input_sequence-1]  # input_sequenceあるので最後を取得
            #v_val = np.transpose(v_val, (1, 0))  # (w,h) -> (h,w)

            img = self._saliency_map(v_val, org_img, org_shape)
            imgs.append(img)
            names.append("value")
        

        # plot
        for i in range(len(imgs)):
            plt.subplot(2, 3, i+1)
            plt.gca().tick_params(labelbottom="off",bottom="off") # x軸の削除
            plt.gca().tick_params(labelleft="off",left="off") # y軸の削除
            plt.title(names[i]).set_fontsize(12)
            plt.imshow(imgs[i])

    
    def _grad_cam(self, c_output, c_val, org_img, org_shape):
        if self.agent.learner.lstm_type == "":
            c_output = c_output[0]
            c_val = c_val[0]
        else:
            c_output = c_output[0][-1]
            c_val = c_val[0][-1]

        weights = np.mean(c_val, axis=(0, 1))
        cam = np.dot(c_output, weights)

        cam = cv2.resize(cam, org_shape, cv2.INTER_LINEAR)
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        cam = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        rate = 0.3
        cam = cv2.addWeighted(src1=org_img, alpha=(1-rate), src2=cam, beta=rate, gamma=0)
        cam = cv2.cvtColor(cam, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
        return cam

    def _saliency_map(self, c_val, org_img, org_shape):
        img = np.abs(c_val)

        img = cv2.resize(img, org_shape, cv2.INTER_LINEAR)
        img = np.maximum(img, 0)
        img = img / img.max()
        img = cv2.applyColorMap(np.uint8(255 * img), cv2.COLORMAP_JET)
        rate = 0.4
        img = cv2.addWeighted(src1=org_img, alpha=(1-rate), src2=img, beta=rate, gamma=0)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 色をRGBに変換
        return img


class LoggerType(enum.Enum):
    STEP = 0
    TIME = 1

class TimeStop(keras.callbacks.Callback):
    def __init__(self, second):
        self.second = second

    def on_train_begin(self, logs={}):
        self.t0 = time.time()
    
    def on_epoch_end(self, epoch, logs={}):
        if time.time() - self.t0 > self.second:
            raise KeyboardInterrupt()

class TrainLogger(keras.callbacks.Callback):
    def __init__(self,
            logger_type,
            interval,
            savefile,
            test_agent=None,
            test_env=None,
            test_episodes=10,
            test_save_max_reward_file="",
            verbose=1
        ):
        self.logger_type = logger_type
        self.savefile = savefile
        self.interval = interval
        self.test_env = test_env
        self.test_agent = test_agent
        self.test_episodes = test_episodes
        self.test_save_max_reward_file = test_save_max_reward_file
        self.verbose = verbose

        self.max_reward_file = ""
        self.max_reward_test = None


    def _init(self):
        self.rewards = []
        self.count = 0


    def _is_record(self):
        if self.logger_type == LoggerType.TIME:
            if time.time() - self.t1 < self.interval:
                return False
            self.t1 = time.time()
        elif self.logger_type == LoggerType.STEP:
            if self.step < self.interval:
                return False
            self.step = 0
        return True
    
    def _record(self, logs):
        if logs is None:
            logs = {}
        if len(self.rewards) == 0:
            self.rewards = [0]

        rewards = np.asarray(self.rewards)
        d = {
            "time": time.time() - self.t0,
            "reward_min": float(rewards.min()),
            "reward_ave": float(rewards.mean()),
            "reward_max": float(rewards.max()),
            "count": self.count,
            "nb_steps": int(logs.get("nb_steps", 0)),
        }

        if self.test_agent is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                name = os.path.join(tmpdir, "tmp")
                self.model.save_weights(name, overwrite=True)
                self.test_agent.load_weights(name)

            history = self.test_agent.test(self.test_env, nb_episodes=self.test_episodes, visualize=False, verbose=False)
            rewards = np.asarray(history.history["episode_reward"])
            mean_reward = rewards.mean()
            if self.test_save_max_reward_file != "":
                if self.max_reward_test is None or self.max_reward_test < mean_reward:
                    self.max_reward_test = mean_reward
                    self.max_reward_file = self.test_save_max_reward_file.format(step=d["nb_steps"],reward=mean_reward)
                    self.model.save_weights(self.max_reward_file, overwrite=True)
                    if self.verbose > 0:
                        print("weight save, ave reward:{:.4f}".format(mean_reward))

            d["test_reward_min"] = float(rewards.min())
            d["test_reward_ave"] = float(mean_reward)
            d["test_reward_max"] = float(rewards.max())

        if self.verbose > 0:
            m = d["time"] / 60.0
            s = "Steps {}, Time: {:.2f}m, ".format(d["nb_steps"], m)
            if "test_reward_min" in d:
                s += "TestReward: {:6.2f} - {:6.2f} (ave: {:6.2f}), ".format(
                    d["test_reward_min"],
                    d["test_reward_max"],
                    d["test_reward_ave"],
                )
            s += "Reward: {:6.2f} - {:6.2f} (ave: {:6.2f})".format(
                d["reward_min"],
                d["reward_max"],
                d["reward_ave"],
            )
            print(s)

        self._init()

        # add file
        if self.savefile != "":
            s = json.dumps(d)
            with open(self.savefile, "a") as f:
                f.write("{}\n".format(s))


    def on_train_begin(self, logs={}):
        if os.path.isfile(self.savefile):
            os.remove(self.savefile)
        self.t0 = self.t1 = time.time()
        self.step = 0
        self._init()
        self._record(logs)

    def on_train_end(self, logs={}):
        if self.verbose > 0:
            print("done, took {:.3f} minutes".format((time.time() - self.t0)/60.0))

        self._record(logs)
    
    def on_step_end(self, batch, logs={}):
        if self.logger_type == LoggerType.STEP:
            self.step += 1


    def on_episode_end(self, episode, logs={}):
        self.rewards.append(logs["episode_reward"])
        self.count += 1

        if not self._is_record():
            return
        self._record(logs)
    
    #--------------------

    def getLogs(self):
        logs = []

        if not os.path.isfile(self.savefile):
            return []

        with open(self.savefile, "r") as f:
            for line in f:
                d = json.loads(line)
                logs.append(d)

        return logs


    def drawGraph(self, base="time"):
        
        log_x = []
        log_ax2_y = []
        log_y1 = []
        log_y2 = []
        log_y3 = []
        label = ""
        for log in self.getLogs():
            if log["nb_steps"] == 0:
                continue
            if base == "time":
                log_x.append(log["time"]/60.0)
                log_ax2_y.append(log["nb_steps"])
            else:
                log_x.append(log["nb_steps"])
                log_ax2_y.append(log["time"]/60.0)
            if "test_reward_min" in log:
                label = "test reward"
                log_y1.append(log["test_reward_min"])
                log_y2.append(log["test_reward_ave"])
                log_y3.append(log["test_reward_max"])
            else:
                label = "reward"
                log_y1.append(log["reward_min"])
                log_y2.append(log["reward_ave"])
                log_y3.append(log["reward_max"])

        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot(log_x, log_y1, marker="o", label="min")
        ax1.plot(log_x, log_y2, marker="o", label="ave")
        ax1.plot(log_x, log_y3, marker="o", label="max")

        ax2 = ax1.twinx()
        ax2.plot(log_x, log_ax2_y, color="black", linestyle="dashed")

        ax1.grid(True)
        ax1.legend()
        ax1.set_ylabel(label)
        if base == "time":
            ax1.set_xlabel("Time(m)")
            ax2.set_ylabel("Steps")
        else:
            ax1.set_xlabel("Steps")
            ax2.set_ylabel("Time(m)")

        plt.show()



# copy from: https://github.com/keras-rl/keras-rl/blob/master/rl/callbacks.py
class ModelIntervalCheckpoint(rl.callbacks.Callback):
    def __init__(self, filepath, interval, save_memory=False, verbose=0):
        super(ModelIntervalCheckpoint, self).__init__()
        self.filepath = filepath
        self.interval = interval
        self.verbose = verbose
        self.save_memory = save_memory
        self.total_steps = 0

    def on_step_end(self, step, logs={}):
        """ Save weights at interval steps during training """
        self.total_steps += 1
        if self.total_steps % self.interval != 0:
            # Nothing to do.
            return

        filepath = self.filepath.format(step=self.total_steps, **logs)
        if self.verbose > 0:
            print('Step {}: saving model to {}'.format(
                self.total_steps, filepath))
        self.model.save_weights(filepath, overwrite=True, save_memory=self.save_memory)





class DisSaveManager(DisCallback):
    def __init__(self,
            save_dirpath,
            is_load=False,
            save_overwrite=True,
            save_memory=False,
            checkpoint=False,
            checkpoint_interval=10000,
            verbose=1,
        ):
        self.save_dirpath = save_dirpath
        self.is_load = is_load
        self.save_overwrite = save_overwrite
        self.save_memory = save_memory
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.verbose = verbose

    def on_dis_learner_begin(self, learner):
        if not self.is_load:
            return
        path = os.path.join(self.save_dirpath, "last", "learner.dat")
        if self.verbose > 0:
            print("load: {}".format(path))
        learner.load_weights(path, load_memory=self.save_memory)

    def on_dis_learner_train_end(self, learner):
        if not self.checkpoint:
            return
        n = learner.train_count.value
        if (n+1) % self.checkpoint_interval == 0:
            dirname = self._get_checkpoint_dir(n+1)
            path = os.path.join(dirname, "learner.dat")
            if self.verbose > 0:
                print("save: {}".format(path))
            learner.save_weights(path, overwrite=True, save_memory=self.save_memory)

    def on_dis_learner_end(self, learner):
        dirname = os.path.join(self.save_dirpath, "last")
        os.makedirs(dirname, exist_ok=True)
        path = os.path.join(dirname, "learner.dat")
        if self.verbose > 0:
            print("save: {}".format(path))
        learner.save_weights(path, self.save_overwrite, save_memory=self.save_memory)

    def on_dis_actor_begin(self, index, actor):
        self.actor = actor
        if not self.is_load:
            return
        path = os.path.join(self.save_dirpath, "last", "actor{}.dat".format(index))
        if self.verbose > 0:
            print("load: {}".format(path))
        actor.load_weights(path)

    def on_dis_actor_end(self, index, actor):
        dirname = os.path.join(self.save_dirpath, "last")
        os.makedirs(dirname, exist_ok=True)
        path = os.path.join(dirname, "actor{}.dat".format(index))
        if self.verbose > 0:
            print("save: {}".format(path))
        actor.save_weights(path, self.save_overwrite)

    def on_step_end(self, episode, logs={}):
        if not self.checkpoint:
            return
        n = self.actor.train_count.value
        if (n+1) % self.checkpoint_interval == 0:
            dirname = self._get_checkpoint_dir(n+1)
            path = os.path.join(dirname, "actor{}.dat".format(self.actor.actor_index))
            if self.verbose > 0:
                print("save: {}".format(path))
            self.actor.save_weights(path, overwrite=True)

    def _get_learner_path(self):
        return os.path.join(self.save_dirpath, "last", "learner.dat")

    def _get_checkpoint_dir(self, count):
        dirname = os.path.join(self.save_dirpath, "checkpoint_{}".format(count))
        os.makedirs(dirname, exist_ok=True)
        path = os.path.join(dirname)
        return path


class DisTrainLogger(DisCallback):
    def __init__(self,
            logger_type,
            interval,
            savedir,
            test_actor=None,
            test_env=None,
            test_episodes=10,
            test_save_max_reward_file="",
            verbose=1
        ):
        self.logger_type = logger_type
        self.savedir = savedir
        self.interval = interval
        self.test_env = test_env
        self.test_actor = test_actor
        self.test_episodes = test_episodes
        self.test_save_max_reward_file = test_save_max_reward_file
        self.verbose = verbose

        self.max_reward_file = ""
        self.max_reward_test = None
        
    def _add_logfile(self, filename, data):
        path = os.path.join(self.savedir, filename)
        with open(path, "a") as f:
            f.write("{}\n".format(json.dumps(data)))

    def _is_record(self):
        if self.logger_type == LoggerType.TIME:
            if time.time() - self.t1 < self.interval:
                return False
            self.t1 = time.time()
        elif self.logger_type == LoggerType.STEP:
            if self.step == 0:
                return False
            if self.step < self.next_step:
                return False
            self.next_step += self.interval

        return True

    def on_dis_train_begin(self):
        os.makedirs(self.savedir, exist_ok=True)
        for fn in glob.glob(os.path.join(self.savedir, "*.json")):
            os.remove(fn)
        self.t0 = self.t1 = time.time()
        self.step = 0
        self.next_step = 0

    def on_dis_train_end(self):
        if self.verbose > 0:
            print("done, took {:.3f} minutes".format((time.time() - self.t0)/60.0))

    #--- learner ---

    def _record_learner(self, learner):
        d = {
            "name": "learner",
            "time": time.time() - self.t0,
            "train_count": learner.learner.train_count,
        }
        if self.test_actor is not None:
            with tempfile.TemporaryDirectory() as tmpdir:
                name = os.path.join(tmpdir, "tmp")
                learner.save_weights(name, overwrite=True)
                test_agent = Agent57.createTestAgentStatic(learner.kwargs, self.test_actor, name)
            env = self.test_env()
            history = test_agent.test(env, nb_episodes=self.test_episodes, visualize=False, verbose=False)
            env.close()
            rewards = np.asarray(history.history["episode_reward"])

            mean_reward = rewards.mean()
            if self.test_save_max_reward_file != "":
                if self.max_reward_test is None or self.max_reward_test < mean_reward:
                    self.max_reward_test = mean_reward
                    self.max_reward_file = self.test_save_max_reward_file.format(step=learner.learner.train_count,reward=mean_reward)
                    learner.save_weights(self.max_reward_file, overwrite=True)
                    if self.verbose > 0:
                        print("weight save, ave reward:{:.4f}".format(mean_reward))


            d["test_reward_min"] = float(rewards.min())
            d["test_reward_ave"] = float(mean_reward)
            d["test_reward_max"] = float(rewards.max())
        else:
            d["test_reward_min"] = 0
            d["test_reward_ave"] = 0
            d["test_reward_max"] = 0
        
        self._add_logfile("learner.json", d)
        if self.verbose > 0:
            m = d["time"] / 60.0
            print("{:8} Train {}, Time: {:.2f}m, TestReward: {:7.2f} - {:7.2f} (ave: {:7.2f})".format(
                d["name"],
                d["train_count"],
                m,
                d["test_reward_min"],
                d["test_reward_max"],
                d["test_reward_ave"]))
    

    def on_dis_learner_train_end(self, learner):
        self.step = learner.learner.train_count
        if not self._is_record():
            return
        self._record_learner(learner)

    def on_dis_learner_end(self, learner):
        self.step = learner.learner.train_count
        self._record_learner(learner)

    #--- actor ---

    def _actor_init(self):
        self.rewards = []
        self.actor_count = 0

    def _record_actor(self, index, logs={}):
        if len(self.rewards) == 0:
            self.rewards = [0]
        rewards = np.asarray(self.rewards)
        d = {
            "name": "actor{}".format(index),
            "time": time.time() - self.t0,
            "reward_min": float(rewards.min()),
            "reward_ave": float(rewards.mean()),
            "reward_max": float(rewards.max()),
            "count": self.actor_count,
            "train_count": int(self.actor.train_count.value),
            "nb_steps": int(logs.get("nb_steps", 0)),
        }
        self._actor_init()

        if self.verbose > 0:
            m = d["time"] / 60.0
            print("{:8} Train {}, Time: {:.2f}m, Reward    : {:7.2f} - {:7.2f} (ave: {:7.2f}), nb_steps: {}".format(
                d["name"],
                d["train_count"],
                m,
                d["reward_min"],
                d["reward_max"],
                d["reward_ave"],
                d["nb_steps"]))
        self._add_logfile("actor{}.json".format(self.actor_index), d)

    def on_dis_actor_begin(self, index, actor):
        self.actor_index = index
        self.actor = actor
    
    def on_dis_actor_end(self, index, actor):
        self._record_actor(index)

    def on_train_begin(self, logs={}):
        self._actor_init()

    def on_episode_end(self, episode, logs={}):
        self.rewards.append(logs["episode_reward"])
        self.actor_count += 1
        self.step = int(self.actor.train_count.value)
        
        if not self._is_record():
            return
        self._record_actor(self.actor_index, logs)

    #--- other ---

    def getLogs(self):
        logs = []

        for fn in glob.glob(os.path.join(self.savedir, "*.json")):
            with open(fn, "r") as f:
                for line in f:
                    d = json.loads(line)
                    logs.append(d)

        return logs

    def drawGraph(self, base="time", actors=-1):
        
        learner_logs = {
            "x": [], 
            "ax2_y": [], 
            "y1": [], 
            "y2": [], 
            "y3": [], 
        }
        actors_logs = {}
        x_max = 0
        for log in self.getLogs():
            name = log["name"]

            t = log["time"]/60.0
            if base == "time":
                if x_max < t:
                    x_max = t
            else:
                if x_max < log["train_count"]:
                    x_max = log["train_count"]
            
            if name == "learner":
                if base == "time":
                    learner_logs["x"].append(t)
                    learner_logs["ax2_y"].append(log["train_count"])
                else:
                    learner_logs["x"].append(log["train_count"])
                    learner_logs["ax2_y"].append(t)
                learner_logs["y1"].append(log["test_reward_min"])
                learner_logs["y2"].append(log["test_reward_ave"])
                learner_logs["y3"].append(log["test_reward_max"])

            else:
                if name not in actors_logs:
                    actors_logs[name] = {
                        "x": [], 
                        "ax2_y": [], 
                        "y1": [], 
                        "y2": [], 
                        "y3": [], 
                    }
                if base == "time":
                    actors_logs[name]["x"].append(t)
                    actors_logs[name]["ax2_y"].append(log["train_count"])
                else:
                    actors_logs[name]["x"].append(log["train_count"])
                    actors_logs[name]["ax2_y"].append(t)

                actors_logs[name]["y1"].append(log["reward_min"])
                actors_logs[name]["y2"].append(log["reward_ave"])
                actors_logs[name]["y3"].append(log["reward_max"])
        
        if actors == -1:
            n = len(actors_logs)+1
        else:
            n = actors + 1

        # learner
        fig = plt.figure()
        ax1 = fig.add_subplot(n, 1, 1)
        ax2 = ax1.twinx()
        ax2.plot(learner_logs["x"], learner_logs["ax2_y"], color="black", linestyle="dashed")
        ax1.plot(learner_logs["x"], learner_logs["y1"], marker="o", label="min")
        ax1.plot(learner_logs["x"], learner_logs["y2"], marker="o", label="ave")
        ax1.plot(learner_logs["x"], learner_logs["y3"], marker="o", label="max")
        #if x_max > 0:
        #    ax1.set_xlim([0, x_max])
        ax1.grid(True)
        ax1.legend()
        if base == "time":
            ax1.set_title("Time(m)")
            ax2.set_ylabel("TrainCount")
        else:
            ax1.set_title("TrainCount")
            ax2.set_ylabel("Time(m)")
        ax1.set_ylabel("Learner")

        # actors
        for i in range(n-1):
            name = "actor{}".format(i)
            v = actors_logs[name]

            ax1 = fig.add_subplot(n, 1, 2+i)
            ax2 = ax1.twinx()
            ax2.plot(v["x"], v["ax2_y"], color="black", linestyle="dashed")
            ax1.plot(v["x"], v["y1"], marker="o", label="min")
            ax1.plot(v["x"], v["y2"], marker="o", label="ave")
            ax1.plot(v["x"], v["y3"], marker="o", label="max")
            #if x_max > 0:
            #    ax1.set_xlim([0, x_max])
            ax1.grid(True)
            ax1.set_ylabel(name)
        
        plt.show()



