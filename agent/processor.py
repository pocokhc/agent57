import rl.core
from PIL import Image, ImageDraw
import numpy as np

import math


class PendulumProcessorForDQN(rl.core.Processor):
    """
    https://github.com/openai/gym/wiki/Pendulum-v0
    """

    def __init__(self, 
            reward_clip=(-0.5, 1),   # None is disable
            action_division=5,
            enable_image=False, 
            image_shape=(84,84)
        ):
        self.reward_clip = reward_clip
        self.enable_image = enable_image
        self.image_shape = image_shape
        self.nb_actions = action_division

        # -2 ～ 2 で分割する
        self.actid_to_value = {}
        for i in range(action_division):
            n = (4.0/(action_division-1))*i - 2.0
            self.actid_to_value[i] = [n]
        
        # 最低報酬
        theta = math.pi
        theta_dt = 8
        action = 2
        self.reward_low = -(theta**2 + 0.1*theta_dt**2 + 0.001*action**2)
        self.reward_high = 0

    def process_observation(self, observation):
        if not self.enable_image:
            return observation
        return self._get_rgb_state(observation)  # reshazeせずに返す
        
    def process_action(self, action):
        return self.actid_to_value[action]

    def get_keys_to_action(self):
        return {
            ():2,
            (ord('z'),):0,
            (ord('x'),):1,
            (ord('c'),):3,
            (ord('v'),):4,
        }

    def process_reward(self, reward):
        if self.reward_clip is None:
            return reward

        # min max normarization
        reward = ((reward - self.reward_low) / (self.reward_high - self.reward_low))*(self.reward_clip[1] - self.reward_clip[0]) + self.reward_clip[0]
        return reward

    # 状態（x,y座標）から対応画像を描画する関数
    def _get_rgb_state(self, state):
        img_size = self.image_shape[0]
        h_size = img_size/2.0

        img = Image.new("RGB", self.image_shape, (255, 255, 255))
        dr = ImageDraw.Draw(img)

        # 棒の長さ
        l = img_size/4.0 * 3.0/ 2.0

        # 棒のラインの描写
        dr.line(((h_size - l * state[1], h_size - l * state[0]), (h_size, h_size)), (0, 0, 0), 1)

        # 棒の中心の円を描写（それっぽくしてみた）
        buff = img_size/32.0
        dr.ellipse(((h_size - buff, h_size - buff), (h_size + buff, h_size + buff)), 
                   outline=(0, 0, 0), fill=(255, 0, 0))

        # 画像の一次元化（GrayScale化）とarrayへの変換
        pilImg = img.convert("L")
        img_arr = np.asarray(pilImg)

        # 画像の規格化
        img_arr = img_arr/255.0

        return img_arr



class CartPoleProcessor(rl.core.Processor):
    """
    https://github.com/openai/gym/wiki/CartPole-v0
    """

    def __init__(self, enable_reward_step=False):
        self.enable_reward_step = enable_reward_step
        self.step = 0
        
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        
        if not self.enable_reward_step:
            return observation, reward, done, info
        
        self.step += 1
         
        if done :
            if self.step > 195:
                reward = 1
            else:
                reward = -1
            self.step = 0
        else:
            reward = 0

        return observation, reward, done, info

    def get_keys_to_action(self):
        return {
            (ord('a'),):0,
            (ord('d'),):1,
        }



class MountainCarProcessor(rl.core.Processor):
    def __init__(self, enable_reward_step=False):
        self.enable_reward_step = enable_reward_step
        self.step = 0
        
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)
        
        if not self.enable_reward_step:
            return observation, reward, done, info
        
        self.step += 1
         
        if done :
            if self.step > 195:
                reward = 1
            else:
                reward = -1
            self.step = 0
        else:
            reward = 0

        return observation, reward, done, info

    def get_keys_to_action(self):
        return {
            (ord('a'),):0,
            (ord('d'),):1,
        }


class AcrobotProcessor(rl.core.Processor):
    def __init__(self, enable_reward_step=False):
        self.enable_reward_step = enable_reward_step
        self.step = 0

    
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)

        if not self.enable_reward_step:
            return observation, reward, done, info
        
        self.step += 1
        if done :
            reward = 500-self.step
            self.step = 0

        else:
            reward = 0

        return observation, reward, done, info


class AtariProcessor(rl.core.Processor):
    def __init__(self,
            reshape_size=(84, 84),
            enable_clip=False,
            max_steps=0,
            freeze_check=0,
            no_reward_check=0,
            penalty_reward=0,
        ):
        self.image_shape = reshape_size
        self.enable_clip = enable_clip
        self.max_steps = max_steps
        self.freeze_check = freeze_check
        self.no_reward_check = no_reward_check
        self.penalty_reward = penalty_reward
        self._init()
        
    def _init(self):
        self.step = 0
        if self.freeze_check > 0:
            self.recent_observations = [
                np.zeros(self.image_shape) for _ in range(self.freeze_check)
            ]
        if self.no_reward_check > 0:
            self.no_reward_count = 0
            
        
    def process_step(self, observation, reward, done, info):
        observation = self.process_observation(observation)
        reward = self.process_reward(reward)
        info = self.process_info(info)

        # freeze check
        if self.freeze_check > 0:
            self.recent_observations.pop(0)
            self.recent_observations.append(observation)
            # 全部同じ画像なら freeze 判定
            f = True
            for i in range(self.freeze_check-1):
                if not np.array_equal(self.recent_observations[i], self.recent_observations[i+1]):
                    f = False
                    break
            if f:
                self._init()
                done = True
                reward = self.penalty_reward
        
        # no_reward_check
        if self.no_reward_check > 0:
            if reward == 0:
                self.no_reward_count += 1
            else:
                self.no_reward_count = 0
            if self.no_reward_count > self.no_reward_check:
                self._init()
                done = True
                reward = self.penalty_reward

        # step制限確認
        if self.max_steps > 0:
            self.step += 1
            if self.step >= self.max_steps:
                self._init()
                done = True

        return observation, reward, done, info

    def process_action(self, action):
        return action

    def process_observation(self, observation):
        img = Image.fromarray(observation)
        img = img.resize(self.image_shape).convert('L')  # resize and convert to grayscale
        return np.array(img) / 255

    def process_reward(self, reward):
        if self.enable_clip:
            return np.clip(reward, -1., 1.)
        return reward

class AtariBreakout(AtariProcessor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n = 0
        self.nb_actions = 3

    def process_action(self, action):
        self.n += 1
        if self.n % 10 == 0:
            return 1
        keys = [0, 2, 3]
        return keys[action]

    def get_keys_to_action(self):
        return {
            ():0,
            (ord('d'),):1,
            (ord('a'),):2,
        }


class AtariPong(AtariProcessor):
    def __init__(self, end_count=20, **kwargs):
        super().__init__(**kwargs)
        self.nb_actions = 3
        self.end_count = end_count
        self.total_count = 0


    def process_action(self, action):
        keys = [0, 2, 3]
        return keys[action]

    def process_step(self, observation, reward, done, info):
        observation, reward, done, info = super().process_step(observation, reward, done, info)
        if reward != 0:
            self.total_count += 1
        if self.total_count == self.end_count:
            done = True
        if done:
            self.total_count = 0
        return observation, reward, done, info

    def get_keys_to_action(self):
        return {
            ():0,
            (ord('a'),):1,
            (ord('d'),):2,
        }


