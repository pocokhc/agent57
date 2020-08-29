from keras.models import Model
from keras.layers import Input, Flatten, Permute, TimeDistributed, LSTM, Dense, Concatenate, Reshape, Lambda, Conv2D, MaxPooling2D, Activation, Add
import tensorflow as tf
from keras import backend as K
import numpy as np

import enum

def clipped_error_loss(y_true, y_pred):
    err = y_true - y_pred  # エラー
    L2 = 0.5 * K.square(err)
    L1 = K.abs(err) - 0.5

    # エラーが[-1,1]区間ならL2、それ以外ならL1を選択する。
    loss = tf.where((K.abs(err) < 1.0), L2, L1)   # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)


class InputType(enum.Enum):
    VALUES = 1    # 画像無し
    GRAY_2ch = 3  # (width, height)
    GRAY_3ch = 4  # (width, height, 1)
    COLOR = 5     # (width, height, ch)

class DuelingNetwork(enum.Enum):
    AVERAGE = 0
    MAX = 1
    NAIVE = 2

class LstmType(enum.Enum):
    NONE = 0
    STATELESS = 1
    STATEFUL = 2

class UvfaType(enum.Enum):
    ACTION = 1
    REWARD_EXT = 2
    REWARD_INT = 3
    POLICY = 4


class ModelBuilder():
    def __init__(self,
            input_shape,
            input_type,
            input_model,
            input_model_emb,
            input_model_rnd,
            batch_size,
            nb_actions,
            input_sequence,
            enable_dueling_network,
            dueling_network_type,
            dense_units_num,
            lstm_type,
            lstm_units_num,
            policy_num,
        ):
        self.input_shape = input_shape
        self.input_type = input_type
        self.input_model = input_model
        self.input_model_emb = input_model_emb
        self.input_model_rnd = input_model_rnd
        self.batch_size = batch_size
        self.nb_actions = nb_actions
        self.input_sequence = input_sequence
        self.enable_dueling_network = enable_dueling_network
        self.dueling_network_type = dueling_network_type
        self.dense_units_num = dense_units_num
        self.lstm_type = lstm_type
        self.lstm_units_num = lstm_units_num
        self.policy_num = policy_num

        if self.input_model_emb is None:
            self.input_model_emb = self.input_model
        if self.input_model_rnd is None:
            self.input_model_rnd = self.input_model

        if input_type == InputType.GRAY_2ch or input_type == InputType.GRAY_3ch or input_type == InputType.COLOR:
            assert self.input_model is not None

            # 画像入力の制約
            # LSTMを使う場合: 画像は(w,h,ch)で入力できます。
            # LSTMを使わない場合：
            #   input_sequenceが1：全て使えます。
            #   input_sequenceが1以外：GRAY_2ch のみ使えます。
            if lstm_type == LstmType.NONE and input_sequence != 1:
                assert (input_type == InputType.GRAY_2ch), "input_iimage can use GRAY_2ch."
    
    def _build_input_layer(self):

        if self.input_type == InputType.VALUES:
            if self.lstm_type != LstmType.STATEFUL:
                c = input_ = Input(
                    shape=(self.input_sequence,) + self.input_shape)
            else:
                c = input_ = Input(
                    batch_shape=(self.batch_size, self.input_sequence) + self.input_shape)
        elif self.input_type == InputType.GRAY_2ch:
            if self.lstm_type != LstmType.STATEFUL:
                c = input_ = Input(
                    shape=(self.input_sequence,) + self.input_shape)
            else:
                c = input_ = Input(
                    batch_shape=(self.batch_size, self.input_sequence) + self.input_shape)
        else:
            if self.lstm_type != LstmType.STATEFUL:
                c = input_ = Input(shape=self.input_shape)
            else:
                c = input_ = Input(
                    batch_shape=(self.batch_size, self.input_sequence) + self.input_shape)

        return c, input_

    def _build_image_layer(self, input_model, lstm_type, c1, c2=None):
        if self.input_type == InputType.VALUES:
            if lstm_type == LstmType.NONE:
                c1 = Flatten()(c1)
                if c2 is not None:
                    c2 = Flatten()(c2)
            else:
                c1 = TimeDistributed(Flatten())(c1)
                if c2 is not None:
                    c2 = TimeDistributed(Flatten())(c2)
        else:
            if lstm_type == LstmType.NONE:
                
                if self.input_type == InputType.GRAY_2ch:
                    # (input_seq, w, h) ->(w, h, input_seq)
                    c1 = Permute((2, 3, 1))(c1)
                    if c2 is not None:
                        c2 = Permute((2, 3, 1))(c2)
            elif lstm_type == LstmType.STATELESS or lstm_type == LstmType.STATEFUL:
                is_lstm = True
                if self.input_type == InputType.GRAY_2ch:
                    # (time steps, w, h) -> (time steps, w, h, ch)
                    c1 = Reshape((self.input_sequence, ) + self.input_shape + (1,) )(c1)
                    if c2 is not None:
                        c2 = Reshape((self.input_sequence, ) + self.input_shape + (1,) )(c2)
            else:
                raise ValueError('lstm_type is undefined: {}'.format(lstm_type))
        if input_model is not None:
            is_lstm = (lstm_type != LstmType.NONE)
            c1, c2 = input_model.create_input_model(c1, is_lstm, c2)

        if c2 is None:
            return c1
        else:
            return c1, c2


    def build_actval_func_model(self, optimizer, uvfa):
        c, input_ = self._build_input_layer()
        c = self._build_image_layer(self.input_model, self.lstm_type, c)
        
        # uvfa
        uvfa_input_num = 0
        if UvfaType.ACTION in uvfa:
            uvfa_input_num += self.nb_actions
        if UvfaType.REWARD_EXT in uvfa:
            uvfa_input_num += 1
        if UvfaType.REWARD_INT in uvfa:
            uvfa_input_num += 1
        if UvfaType.POLICY in uvfa:
            uvfa_input_num += self.policy_num
        if len(uvfa) > 0:
            if self.lstm_type == LstmType.NONE:
                c2 = input2 = Input(shape=(uvfa_input_num,))
            elif self.lstm_type == LstmType.STATELESS:
                c2 = input2 = Input(shape=(self.input_sequence, uvfa_input_num,))
            else:
                c2 = input2 = Input(batch_shape=(self.batch_size, self.input_sequence, uvfa_input_num))
            c = Concatenate()([c, c2])
        
        # lstm layer
        if self.lstm_type == LstmType.STATELESS:
            c = LSTM(self.lstm_units_num, name="lstm")(c)
        elif self.lstm_type == LstmType.STATEFUL:
            c = LSTM(self.lstm_units_num, stateful=True, name="lstm")(c)
        else:
            c = Dense(self.dense_units_num, activation="relu")(c)

        # dueling network
        if self.enable_dueling_network:
            # value
            v = Dense(self.dense_units_num, activation="relu")(c)
            v = Dense(1, name="v")(v)

            # advance
            adv = Dense(self.dense_units_num, activation='relu')(c)
            adv = Dense(self.nb_actions, name="adv")(adv)

            # 連結で結合
            c = Concatenate()([v,adv])
            if self.dueling_network_type == DuelingNetwork.AVERAGE:
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.mean(a[:, 1:], axis=1, keepdims=True), output_shape=(self.nb_actions,))(c)
            elif self.dueling_network_type == DuelingNetwork.MAX:
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] - K.max(a[:, 1:], axis=1, keepdims=True), output_shape=(self.nb_actions,))(c)
            elif self.dueling_network_type == DuelingNetwork.NAIVE:
                c = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:], output_shape=(self.nb_actions,))(c)
            else:
                raise ValueError('dueling_network_type is undefined')
        else:
            c = Dense(self.dense_units_num, activation="relu")(c)
            c = Dense(self.nb_actions, activation="linear", name="adv")(c)
        
        if len(uvfa) > 0:
            model = Model([input_, input2], c)
        else:
            model = Model(input_, c)
        
        if optimizer is not None:
            model.compile(loss=clipped_error_loss, optimizer=optimizer)

        return model


    def build_embedding_model(self):
        c, input_ = self._build_input_layer()
        c = self._build_image_layer(self.input_model_emb, LstmType.NONE, c)

        c = Dense(32, activation="relu", name="emb_dense")(c)

        model = Model(input_, c)
        # 学習しないので compile は不要
        return model


    def build_embedding_model_train(self, optimizer):
        c1, input1 = self._build_input_layer()
        c2, input2 = self._build_input_layer()

        c1, c2 = self._build_image_layer(self.input_model_emb, LstmType.NONE, c1, c2)

        d = Dense(32, activation="relu", name="emb_dense")
        c1 = d(c1)
        c2 = d(c2)

        c = Concatenate()([c1, c2])
        c = Dense(128, activation="relu")(c)
        c = Dense(self.nb_actions, activation="softmax")(c)

        model = Model([input1, input2], c)
        model.compile(loss='mean_squared_error', optimizer=optimizer)

        return model

    def sync_embedding_model(self, train_model, target_model):
        if self.input_model_emb is not None:
            for name in self.input_model_emb.get_layer_names():
                train_layer = train_model.get_layer(name)
                target_layer = target_model.get_layer(name)
                target_layer.set_weights(train_layer.get_weights())
        train_layer = train_model.get_layer("emb_dense")
        target_layer = target_model.get_layer("emb_dense")
        target_layer.set_weights(train_layer.get_weights())


    def build_rnd_model(self, optimizer):
        c, input_ = self._build_input_layer()
        c = self._build_image_layer(self.input_model_rnd, LstmType.NONE, c)

        c = Dense(128)(c)

        model = Model(input_, c)
        if optimizer is not None:
            model.compile(loss='mean_squared_error', optimizer=optimizer)

        return model


class InputModel():
    """ Abstract base class for all implemented InputModel. """
    def get_layer_names(self):
        raise NotImplementedError()

    def create_input_model(self, c1, is_lstm, c2=None):
        raise NotImplementedError()
    

class DQNImageModel(InputModel):
    """ native dqn image model
    https://arxiv.org/abs/1312.5602
    """

    def get_layer_names(self):
        return [
            "c1",
            "c2",
            "c3",
        ]

    def create_input_model(self, c1, is_lstm, c2=None):
        
        if not is_lstm:
            l = Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu", name="c1")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
    
            l = Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu", name="c2")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            l = Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu", name="c3")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            c1 = Flatten()(c1)
            if c2 is not None:
                c2 = Flatten()(c2)

        else:
            l = TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), padding="same", activation="relu"), name="c1")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
    
            l = TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), padding="same", activation="relu"), name="c2")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            l = TimeDistributed(Conv2D(64, (3, 3), strides=(1, 1), padding="same", activation="relu"), name="c3")
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

            c1 = TimeDistributed(Flatten())(c1)
            if c2 is not None:
                c2 = TimeDistributed(Flatten())(c2)

        return c1, c2


class R2D3ImageModel(InputModel):
    """ R2D3 image model
    https://arxiv.org/abs/1909.01387
    """

    def __init__(self):
        self.names = []

    def get_layer_names(self):
        return self.names

    def create_input_model(self, c1, is_lstm, c2=None):
        self.names = []

        c1, c2 = self._resblock(c1, 16, is_lstm, c2)
        c1, c2 = self._resblock(c1, 32, is_lstm, c2)
        c1, c2 = self._resblock(c1, 32, is_lstm, c2)

        c1 = Activation("relu")(c1)
        if c2 is not None:
            c2 = Activation("relu")(c2)
        
        if not is_lstm:
            c1 = Flatten()(c1)
            if c2 is not None:
                c2 = Flatten()(c2)
        else:
            c1 = TimeDistributed(Flatten())(c1)
            if c2 is not None:
                c2 = TimeDistributed(Flatten())(c2)

        return c1, c2


    def _resblock(self, c1, n_filter, is_lstm, c2):
        if not is_lstm:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = Conv2D(n_filter, (3, 3), strides=(1,1), padding="same", name=n)
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
            l = MaxPooling2D((3, 3), strides=(2,2), padding='same')
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
        else:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = TimeDistributed(Conv2D(n_filter, (3, 3), strides=(1,1), padding="same"), name=n)
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)
            l = TimeDistributed(MaxPooling2D((3, 3), strides=(2,2), padding='same'))
            c1 = l(c1)
            if c2 is not None:
                c2 = l(c2)

        c1, c2 = self._residual_block(c1, n_filter, is_lstm, c2)
        c1, c2 = self._residual_block(c1, n_filter, is_lstm, c2)

        return c1, c2


    def _residual_block(self, c1, n_filter, is_lstm, c2):

        if not is_lstm:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = Conv2D(n_filter, (3, 3), strides=(1,1), padding="same", name=n)
            c1_tmp = Activation("relu")(c1)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)

            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = Conv2D(n_filter, (3, 3), strides=(1,1), padding="same", name=n)
            c1_tmp = Activation("relu")(c1_tmp)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)
        else:
            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = TimeDistributed(Conv2D(n_filter, (3, 3), strides=(1,1), padding="same"), name=n)
            c1_tmp = Activation("relu")(c1)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)

            n = "l{}".format(len(self.names))
            self.names.append(n)
            l = TimeDistributed(Conv2D(n_filter, (3, 3), strides=(1,1), padding="same"), name=n)
            c1_tmp = Activation("relu")(c1_tmp)
            c1_tmp = l(c1_tmp)
            if c2 is not None:
                c2_tmp = Activation("relu")(c2)
                c2_tmp = l(c2_tmp)

        # 結合
        c1 = Add()([c1, c1_tmp])
        if c2 is not None:
            c2 = Add()([c2, c2_tmp])
        return c1, c2



class ValueModel(InputModel):
    def __init__(self, dense_units, layer_num=3):
        self.dense_units = dense_units
        self.layer_num = layer_num

    def get_layer_names(self):
        return [ "l{}".format(i) for i in range(self.layer_num)]

    def create_input_model(self, c1, is_lstm, c2=None):
        
        if not is_lstm:
            for i in range(self.layer_num):
                l = Dense(self.dense_units, activation="relu", name="l{}".format(i))
                c1 = l(c1)
                if c2 is not None:
                    c2 = l(c2)
        else:
            for i in range(self.layer_num):
                l = TimeDistributed(Dense(self.dense_units, activation="relu", name="l{}".format(i)))
                c1 = l(c1)
                if c2 is not None:
                    c2 = l(c2)
        
        return c1, c2

