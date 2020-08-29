# Agemt57(Deep Reinforcement Learning) for Keras-RL
以下Qiita記事の実装コードとなります。
コードの解説については記事を参照してください。

+ [【強化学習】ついに人間を超えた！？Agent57を解説/実装してみた（Keras-RL）](https://qiita.com/pocokhc/items/8684c6c96d3d2963e284)


# 概要
Keras 向けの強化学習ライブラリである [Keras-rl](https://github.com/keras-rl/keras-rl) の Agent を拡張したものとなります。  
以下のアルゴリズムを実装しています。(非公式です)  

- Deep Q Learning(DQN)
- Rainbow
  - Double DQN
  - Priority Experience Reply
  - Dueling Network
  - Multi-Step learning
  - (not implemented Noisy Network)
  - (not implemented Categorical DQN)
- Deep Recurrent Q-Learning(DRQN)
- Ape-X
- Recurrent Replay Distributed DQN(R2D2)
- Recurrent Replay Distributed DQN from Demonstrations(R2D3)
- Agent57
  - Never Give Up(NGU)


# Getting started
## 1. pip install
使っているパッケージは以下です。

+ pip install tensorflow (or tensorflow-cpu or tensorflow-gpu)
+ pip install keras
+ pip install keras-rl
+ pip install gym
+ pip install numpy
+ pip install matplotlib
+ pip install opencv-python
+ pip install pillow
+ pip install pygame

必要に応じて以下のレポジトリも参照してください。

- [OpenAI Gym](https://github.com/openai/gym)
- [Keras-rl](https://github.com/keras-rl/keras-rl)

### 作成時のバージョン

+ windows 10
+ python 3.7.5
+ tensorflow 2.1.0
+ tensorflow-gpu 2.1.0
  + cuda_10.1.243
  + cudnn v7.6.5.32
+ Keras 2.3.1
+ keras-rl 0.4.2
+ gym 0.17.1
+ numpy 1.18.2
+ matplotlib 3.2.1
+ opencv-python 4.1.2.30
+ pillow 6.2.1
+ pygame 1.9.6


## 2. download
このレポジトリをダウンロードします。

``` bash
> git clone https://github.com/pocokhc/agent57.git
```

## 3. Run the program
examples にいくつか実行例が入っています。

``` bash
> cd agent57/examples
> python mountaincar.py
```


# 2020/8 update
UVFA回りを大幅に更新しました。
更新に伴って変更したパラメータがあるので、ここに書いておきます。

1. 入力モデルの変数名を変更
```
image_model
image_model_emb
image_model_rnd
```
↓
```
input_model
input_model_emb
input_model_rnd
```

2. 内部報酬の有効無効から、行動価値関数の有効無効に変更
```
enable_intrinsic_reward
```
↓
```
enable_intrinsic_actval_model
```

3. UVFAの入力項目を設定できるように追加

・入力無し
```
uvfa_ext=[]  # 外部報酬の入力項目
uvfa_int=[]  # 内部報酬の入力項目
```

・入力あり(全部ありの場合)
```
uvfa_ext=[
  UvfaType.ACTION,
  UvfaType.REWARD_EXT,
  UvfaType.REWARD_INT,
  UvfaType.POLICY,
]
uvfa_int=[
  UvfaType.ACTION,
  UvfaType.REWARD_EXT,
  UvfaType.REWARD_INT,
  UvfaType.POLICY,
]
```

4. エピソード終了時に null フレームを追加するオプションを作成

```
enable_add_episode_end_frame=True
```

有効にするとエピソードの最後(terminal=True)のフレームに追加して、
報酬0のnullフレームを経験メモリに追加します。


5. テスト時に実行するポリシーを変更できるオプションを追加
テスト時のpolicyは探索なしの0固定でしたが、これを任意に選択できるようなオプションを追加しました。

```
test_policy = 0
```

