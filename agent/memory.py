import numpy as np

import random
import bisect
import math


class Memory():
    """ Abstract base class for all implemented Memory. """
    def add(self, exp, priority=0):
        raise NotImplementedError()

    def update(self, idx, exp, priority):
        raise NotImplementedError()

    def sample(self, batch_size, steps):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()

    def get_memorys(self):
        raise NotImplementedError()

    def set_memorys(self, data):
        raise NotImplementedError()


class _bisect_wrapper():
    def __init__(self, data, priority):
        self.data = data
        self.priority = priority
    
    def __lt__(self, o):  # a<b
        return self.priority < o.priority


class ReplayMemory(Memory):
    """ ReplayMemory
    https://arxiv.org/abs/1312.5602
    """

    @staticmethod
    def getName():
        return "ReplayMemory"

    def __init__(self, capacity=1_000_000):
        self.capacity = capacity
        self.index = 0
        self.buffer = []

    def add(self, exp, priority=0):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.index] = exp
        self.index = (self.index + 1) % self.capacity

    def update(self, idx, exp, priority):
        pass

    def sample(self, batch_size, steps):
        batchs = random.sample(self.buffer, batch_size)
        indexes = [ 0 for _ in range(batch_size)]
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

    def get_memorys(self):
        return self.buffer[:]

    def set_memorys(self, data):
        for d in data:
            self.add(d)


class PERGreedyMemory(Memory):
    @staticmethod
    def getName():
        return "PERGreedyMemory"

    def __init__(self, capacity=1_000_000):
        self.buffer = []
        self.capacity = capacity
        self.max_priority = 1

    def add(self, exp, priority=0):
        if priority == 0:
            priority = self.max_priority
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は要素を削除
            self.buffer.pop(0)
        
        # priority は最初は最大を選択
        exp = _bisect_wrapper(exp, priority)
        bisect.insort(self.buffer, exp)

    def update(self, idx, exp, priority):
        exp = _bisect_wrapper(exp, priority)
        bisect.insort(self.buffer, exp)

        if self.max_priority < priority:
            self.max_priority = priority
    
    def sample(self, batch_size, step):
        # 取り出す(学習後に再度追加)
        batchs = [self.buffer.pop().data for _ in range(batch_size)]
        indexes = [ 0 for _ in range(batch_size)]
        weights = [ 1 for _ in range(batch_size)]
        return (indexes, batchs, weights)

    def __len__(self):
        return len(self.buffer)

    def get_memorys(self):
        return [(d.data, d.priority) for d in self.buffer]

    def set_memorys(self, data):
        self.buffer = []
        for d in data:
            self.add(d[0], d[1])


class SumTree():
    """
    copy from https://github.com/jaromiru/AI-blog/blob/5aa9f0b/SumTree.py
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.write = 0
        self.tree = [ 0 for _ in range( 2*capacity - 1 )]
        self.data = [ None for _ in range(capacity)]

    def _propagate(self, idx, change):
        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def total(self):
        return self.tree[0]

    def add(self, p, data):
        idx = self.write + self.capacity - 1

        self.data[self.write] = data
        self.update(idx, p)

        self.write += 1
        if self.write >= self.capacity:
            self.write = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1

        return (idx, self.tree[idx], self.data[dataIdx])


class PERProportionalMemory(Memory):
    @staticmethod
    def getName():
        return "PERProportionalMemory"

    def __init__(self,
            capacity=1_000_000,
            alpha=0.6,
            beta_initial=0.4,
            beta_steps=1_000_000,
            enable_is=False
        ):
        self.capacity = capacity
        self.tree = SumTree(capacity)

        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is
        self.alpha = alpha

        self.size = 0
        self.max_priority = 1

    def add(self, exp, priority=0, _alpha_skip=False):
        if priority == 0:
            priority = self.max_priority
        if not _alpha_skip:
            priority = priority ** self.alpha
        self.tree.add(priority, exp)
        self.size += 1
        if self.size > self.capacity:
            self.size = self.capacity

    def update(self, index, exp, priority):
        priority = priority ** self.alpha
        self.tree.update(index, priority)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
            if beta > 1:
                beta = 1
    
        total = self.tree.total()
        for i in range(batch_size):
            
            # indexesにないものを追加
            for _ in range(100):  # for safety
                r = random.random()*total
                (idx, priority, experience) = self.tree.get(r)
                if idx not in indexes:
                    break

            indexes.append(idx)
            batchs.append(experience)

            if self.enable_is:
                # 重要度サンプリングを計算
                weights[i] = (self.size * priority / total) ** (-beta)
            else:
                weights[i] = 1  # 無効なら1

        if self.enable_is:
            # 安定性の理由から最大値で正規化
            weights = weights / weights.max()

        return (indexes ,batchs, weights.tolist())

    def __len__(self):
        return self.size

    def get_memorys(self):
        data = []
        for i in range(self.size):
            d = self.tree.data[i]
            p = self.tree.tree[i+self.capacity-1]
            data.append([d, p])
        
        return data

    def set_memorys(self, data):
        self.tree = SumTree(self.capacity)
        self.size = 0

        for d in data:
            self.add(d[0], d[1], _alpha_skip=True)


def rank_sum(k, a):
    return k*( 2+(k-1)*a )/2

def rank_sum_inverse(k, a):
    if a == 0:
        return k
    t = a-2 + math.sqrt((2-a)**2 + 8*a*k)
    return t/(2*a)

class PERRankBaseMemory(Memory):
    @staticmethod
    def getName():
        return "PERRankBaseMemory"

    def __init__(self,
            capacity=1_000_000,
            alpha=0.6,
            beta_initial=0.4,
            beta_steps=1_000_000,
            enable_is=False
        ):
        self.capacity = capacity
        self.buffer = []
        self.alpha = alpha
        
        self.beta_initial = beta_initial
        self.beta_steps = beta_steps
        self.enable_is = enable_is

        self.max_priority = 1

    def add(self, exp, priority=0):
        if priority == 0:
            priority = self.max_priority
        if self.capacity <= len(self.buffer):
            # 上限より多い場合は要素を削除
            self.buffer.pop(0)
            
        exp = _bisect_wrapper(exp, priority)
        bisect.insort(self.buffer, exp)

    def update(self, index, exp, priority):
        exp = _bisect_wrapper(exp, priority)
        bisect.insort(self.buffer, exp)

        if self.max_priority < priority:
            self.max_priority = priority

    def sample(self, batch_size, step):
        indexes = []
        batchs = []
        weights = np.empty(batch_size, dtype='float32')

        if self.enable_is:
            # βは最初は低く、学習終わりに1にする。
            beta = self.beta_initial + (1 - self.beta_initial) * step / self.beta_steps
            if beta > 1:
                beta = 1

        # 合計値をだす
        buffer_size = len(self.buffer)
        total = rank_sum(buffer_size, self.alpha)
        
        # index_lst
        index_lst = []
        for _ in range(batch_size):

            # index_lstにないものを追加
            for _ in range(100):  # for safety
                r = random.random()*total
                index = rank_sum_inverse(r, self.alpha)
                index = int(index)  # 整数にする(切り捨て)
                if index not in index_lst:
                    index_lst.append(index)
                    break
        
        #assert len(index_lst) == batch_size
        index_lst.sort()

        for i, index in enumerate(reversed(index_lst)):
            o = self.buffer.pop(index)  # 後ろから取得するのでindexに変化なし
            batchs.append(o.data)
            indexes.append(index)

            if self.enable_is:
                # 重要度サンプリングを計算
                # 確率を計算(iでの区間/total)
                r1 = rank_sum(index+1, self.alpha)
                r2 = rank_sum(index, self.alpha)
                priority = (r1-r2) / total
                w = (buffer_size * priority) ** (-beta)
                weights[i] = w
            else:
                weights[i] = 1  # 無効なら1
        
        if self.enable_is:
            if True:
                # 安定性の理由から最大値で正規化
                weights = weights / weights.max()
            else:
                for i in range(len(weights)):
                    if weights[i] > 1:
                        weights[i] = 1

        return (indexes, batchs, weights.tolist())

    def __len__(self):
        return len(self.buffer)

    def get_memorys(self):
        return [(d.data, d.priority) for d in self.buffer]

    def set_memorys(self, data):
        self.buffer = []
        self.max_priority = 1
        for d in data:
            self.add(d[0], d[1])


class EpisodeMemory(Memory):
    def __init__(self, memory, verbose):
        self.max_reward = None
        self.memory = memory
        self.verbose = verbose
    
    def add_episode(self, episode, total_reward):
        if self.memory is None:
            return
        f = True
        if self.max_reward is None:
            self.max_reward = total_reward
        elif self.max_reward <= total_reward:
            f = self.max_reward < total_reward
            self.max_reward = total_reward
        else:
            return
        for e in episode:
            if len(e) == 5:
                self.memory.add(e, e[4])
            else:
                self.memory.add(e)
        if self.verbose > 0 and f:
            print("episode add, reward:{:.4f} length: {} on_memory: {}".format(total_reward, len(episode), len(self.memory)))

    def update(self, idx, exp, priority):
        self.memory.update(idx, exp, priority)

    def sample(self, batch_size, step):
        return self.memory.sample(batch_size, step)

    def __len__(self):
        if self.memory is None:
            return 0
        return len(self.memory)

    def get_memorys(self):
        if self.memory is None:
            return None
        d = {
            "max": self.max_reward,
            "memory": self.memory.get_memorys()
        }
        return d

    def set_memorys(self, data):
        if self.memory is None:
            return None
        self.max_reward = data["max"]
        self.memory.set_memorys(data["memory"])




class MemoryFactory():
    memories = [
        ReplayMemory,
        PERGreedyMemory,
        PERRankBaseMemory,
        PERProportionalMemory,
    ]
    @staticmethod
    def create(name, kwargs):

        for m in MemoryFactory.memories:
            if m.getName() == name:
                return m(**kwargs)
        
        names = []
        for m in MemoryFactory.memories:
            names.append(m.getName())
        raise ValueError('memories is [{}]'.format(",".join(names)))


