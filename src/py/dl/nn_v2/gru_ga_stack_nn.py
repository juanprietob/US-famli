from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import json
import os
import glob
import sys

# class GruAtt(tf.keras.layers.Layer):
#     def __init__(self, units=1024, num_heads=4, key_dim=512, drop_prob=0):
#         super(GruAtt, self).__init__()
        
#         self.drop_prob = drop_prob

#         self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=self.drop_prob)
#         self.proj = layers.Dense(units, activation='relu', use_bias=False)

#     def call(self, x):

#         x_a, w_a = self.att(x, x, return_attention_scores=True)
#         x_a = tf.reduce_sum(x_a, axis=1)
#         x_a = self.proj(x_a)

#         return x_a, w_a

# class BahdanauAttention(tf.keras.layers.Layer):
#     def __init__(self, units):
#         super(BahdanauAttention, self).__init__()
#         self.W1 = tf.keras.layers.Dense(units)
#         self.W2 = tf.keras.layers.Dense(units)
#         self.V = tf.keras.layers.Dense(1)
#         # self.k = k

#     def call(self, query, values):
#         # query hidden state shape == (batch_size, hidden size)
#         # query_with_time_axis shape == (batch_size, 1, hidden size)
#         # values shape == (batch_size, max_len, hidden size)
#         # we are doing this to broadcast addition along the time axis to calculate the score
#         query_with_time_axis = tf.expand_dims(query, 1)

#         # score shape == (batch_size, max_length, 1)
#         # we get 1 at the last axis because we are applying score to self.V
#         # the shape of the tensor before applying self.V is (batch_size, max_length, units)
#         score = self.V(tf.nn.tanh(
#             self.W1(query_with_time_axis) + self.W2(values)))

#         # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
#         # min_score = tf.reshape(min_score, [-1, 1, 1])
#         # score_mask = tf.greater_equal(score, min_score)
#         # score_mask = tf.cast(score_mask, tf.float32)
#         # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

#         # attention_weights shape == (batch_size, max_length, 1)
#         attention_weights = tf.nn.softmax(score, axis=1)

#         # context_vector shape after sum == (batch_size, hidden_size)
#         context_vector = attention_weights * values
#         context_vector = tf.reduce_sum(context_vector, axis=1)

#         return context_vector, attention_weights

# class GruAtt(tf.keras.layers.Layer):
#     def __init__(self, gru_units=512, units=512, num_heads=4, key_dim=512, drop_prob=0):
#         super(GruAtt, self).__init__()
        
#         self.drop_prob = drop_prob

#         self.gru_fwd = layers.GRU(units=gru_units, activation='tanh', dropout=self.drop_prob, return_sequences=True, return_state=True)
#         self.gru_bwd = layers.GRU(units=gru_units, activation='tanh', dropout=self.drop_prob, return_sequences=True, return_state=True, go_backwards=True)

#         self.drop0 = layers.Dropout(self.drop_prob)
#         self.drop1 = layers.Dropout(self.drop_prob)
#         self.drop2 = layers.Dropout(self.drop_prob)
#         self.drop3 = layers.Dropout(self.drop_prob)

#         # self.att_fwd = layers.AdditiveAttention(causal=True, dropout=self.drop_prob)
#         # self.att_bwd = layers.AdditiveAttention(causal=True, dropout=self.drop_prob)

#         # self.att_fwd = layers.Attention(use_scale=False, causal=True, dropout=self.drop_prob)
#         # self.att_bwd = layers.Attention(use_scale=False, causal=True, dropout=self.drop_prob)

#         self.att_fwd = BahdanauAttention(units)
#         self.att_bwd = BahdanauAttention(units)
        
#         self.concat = layers.Concatenate()

#         self.proj = layers.Dense(units, activation='relu', use_bias=False)

class FeedForward(tf.keras.layers.Layer):
    def __init__(self, units_w1=2048, units_w2=512):
        super(FeedForward, self).__init__()

        self.W1 = layers.Conv1D(units_w1, kernel_size=1, activation='relu')
        self.W2 = layers.Conv1D(units_w2, kernel_size=1)

    def call(self, x):

        x = self.W1(x)
        x = self.W2(x)

        return x

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, units=512, num_heads=8, key_dim=64, drop_prob=0):
        super(TransformerBlock, self).__init__()
        
        self.drop_prob = drop_prob

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, dropout=self.drop_prob)
        self.add0 = layers.Add()
        self.layer_norm0 = layers.LayerNormalization()
        self.feed_fwd = FeedForward(units_w1=2048, units_w2=units)

        self.add1 = layers.Add()
        self.layer_norm1 = layers.LayerNormalization()

    def call(self, x):

        x_a, w_a = self.att(x, x, return_attention_scores=True)

        x = self.add0([x, x_a])
        x = self.layer_norm0(x)
        
        x_f = self.feed_fwd(x)

        x = self.add1([x, x_f])
        x = self.layer_norm1(x)

        return x


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, N=3, units=512, num_heads=8, key_dim=64, drop_prob=0):
        super(TransformerEncoder, self).__init__()

        self.transformer_blocks = []
        self.N = N

        for n in range(N):
            self.transformer_blocks.append(TransformerBlock(units=units, num_heads=num_heads, key_dim=key_dim, drop_prob=drop_prob))

    def call(self, x):

        for n in range(self.N):
            x = self.transformer_blocks[n](x)

        return x

class NN(tf.keras.Model):

    def __init__(self, tf_inputs, args):
        super(NN, self).__init__()
        
        learning_rate = args.learning_rate
        decay_steps = args.decay_steps
        decay_rate = args.decay_rate
        staircase = args.staircase
        drop_prob = args.drop_prob

        data_description = tf_inputs.get_data_description()
        self.num_channels = data_description[data_description["data_keys"][0]]["shape"][-1]

        self.num_classes = 2
        self.class_weights_index = -1
        self.enumerate_index = 1

        if "enumerate" in data_description:
            self.enumerate_index = data_description["data_keys"].index(data_description["enumerate"])

            if(data_description[data_description["data_keys"][self.enumerate_index]]["num_class"]):
                self.num_classes = data_description[data_description["data_keys"][self.enumerate_index]]["num_class"]
                print("Number of classes in data description", self.num_classes)
                if "class_weights" in data_description["data_keys"]:
                    self.class_weights_index = data_description["data_keys"].index("class_weights")
                    print("Using weights index", self.class_weights_index)

        self.drop_prob = drop_prob

        self.gru_class = self.make_gru_network()
        self.gru_class.summary()
        
        self.max_value = 290.0
        self.loss = tf.keras.losses.MeanSquaredError()
        # self.loss = SigmoidCrossEntropy(self.max_value)
        # self.loss = tf.keras.losses.MeanAbsoluteError()
        # self.loss = tf.keras.losses.Huber(delta=5.0, reduction=tf.keras.losses.Reduction.SUM)
        # self.loss = tf.keras.losses.LogCosh()
        self.metrics_train = tf.keras.metrics.MeanAbsoluteError()

        if decay_rate != 0.0:
            lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)
        else:
            lr = learning_rate

        self.optimizer = tf.keras.optimizers.Adam(lr)
        

        self.validation_loss = tf.keras.losses.MeanSquaredError()
        # self.validation_loss = tf.keras.losses.Huber(delta=5.0, reduction=tf.keras.losses.Reduction.SUM)
        # self.validation_loss = tf.keras.losses.MeanSquaredError()
        # self.validation_loss = SigmoidCrossEntropy(self.max_value)
        self.validation_metric = tf.keras.metrics.MeanAbsoluteError()

        self.global_validation_metric = float("inf")
        self.global_validation_step = args.in_epoch

    def make_gru_network(self):
        
        x0 = tf.keras.Input(shape=[None, 6, self.num_channels])

        num_channels = self.num_channels

        # # x_spc = tf.math.reduce_max(x0[:,:,:,num_channels:], axis=1)
        # # x_spc = layers.Flatten()(x_spc)
        # # x_spc = layers.Dense(1, activation=None, name='fit_spacing')(x_spc)

        x = layers.Masking(mask_value=-1.0)(x0)
        # x = tf.multiply(x, x_spc)
        # x = layers.BatchNormalization()(x)

        x_0 = layers.Reshape([-1, num_channels])(x[:,:,0,:])
        x_1 = layers.Reshape([-1, num_channels])(x[:,:,1,:])
        x_2 = layers.Reshape([-1, num_channels])(x[:,:,2,:])
        x_3 = layers.Reshape([-1, num_channels])(x[:,:,3,:])
        x_4 = layers.Reshape([-1, num_channels])(x[:,:,4,:])
        x_5 = layers.Reshape([-1, num_channels])(x[:,:,5,:])

        # x_0, w_a0_fwd, w_a0_bwd = GruAtt(units=1024, drop_prob=self.drop_prob)(x_0)
        # x_1, w_a1_fwd, w_a1_bwd = GruAtt(units=1024, drop_prob=self.drop_prob)(x_1)
        # x_2, w_a2_fwd, w_a2_bwd = GruAtt(units=1024, drop_prob=self.drop_prob)(x_2)
        # x_3, w_a3_fwd, w_a3_bwd = GruAtt(units=1024, drop_prob=self.drop_prob)(x_3)
        # x_4, w_a4_fwd, w_a4_bwd = GruAtt(units=1024, drop_prob=self.drop_prob)(x_4)
        # x_5, w_a5_fwd, w_a5_bwd = GruAtt(units=1024, drop_prob=self.drop_prob)(x_5)

        # x_0, w_a0 = GruAtt(drop_prob=self.drop_prob)(x_0)
        # x_1, w_a1 = GruAtt(drop_prob=self.drop_prob)(x_1)
        # x_2, w_a2 = GruAtt(drop_prob=self.drop_prob)(x_2)
        # x_3, w_a3 = GruAtt(drop_prob=self.drop_prob)(x_3)
        # x_4, w_a4 = GruAtt(drop_prob=self.drop_prob)(x_4)
        # x_5, w_a5 = GruAtt(drop_prob=self.drop_prob)(x_5)

        x_0 = TransformerEncoder(units=num_channels, drop_prob=self.drop_prob)(x_0)
        x_1 = TransformerEncoder(units=num_channels, drop_prob=self.drop_prob)(x_1)
        x_2 = TransformerEncoder(units=num_channels, drop_prob=self.drop_prob)(x_2)
        x_3 = TransformerEncoder(units=num_channels, drop_prob=self.drop_prob)(x_3)
        x_4 = TransformerEncoder(units=num_channels, drop_prob=self.drop_prob)(x_4)
        x_5 = TransformerEncoder(units=num_channels, drop_prob=self.drop_prob)(x_5)

        x_0 = layers.GlobalMaxPooling1D()(x_0)
        x_1 = layers.GlobalMaxPooling1D()(x_1)
        x_2 = layers.GlobalMaxPooling1D()(x_2)
        x_3 = layers.GlobalMaxPooling1D()(x_3)
        x_4 = layers.GlobalMaxPooling1D()(x_4)
        x_5 = layers.GlobalMaxPooling1D()(x_5)

        x_0 = tf.expand_dims(x_0, axis=1)
        x_1 = tf.expand_dims(x_1, axis=1)
        x_2 = tf.expand_dims(x_2, axis=1)
        x_3 = tf.expand_dims(x_3, axis=1)
        x_4 = tf.expand_dims(x_4, axis=1)
        x_5 = tf.expand_dims(x_5, axis=1)

        x = layers.Concatenate(axis=1)([x_0, x_1, x_2, x_3, x_4, x_5])
        # x, w_a = GruAtt(units=4096, drop_prob=self.drop_prob)(x)
        x = TransformerEncoder(units=num_channels, drop_prob=self.drop_prob)(x)
        # x_spc = x0[:,0,0,512:513]
        # x = tf.multiply(x, x_spc)
        x = layers.GlobalMaxPooling1D()(x)

        x = layers.Dense(1, activation='sigmoid', name='prediction')(x)
        x = tf.math.add(tf.math.multiply(x, 180.0), 100.0)

        # x_e, x_h_fwd, x_h_bwd = layers.Bidirectional(layers.GRU(units=512, activation='tanh', use_bias=False, kernel_initializer="glorot_normal", dropout=self.drop_prob, return_sequences=True, return_state=True), name="bi_gru0")(x)
        # x_e = layers.Dropout(self.drop_prob)(x_e)
        # x_h_fwd = layers.Dropout(self.drop_prob)(x_h_fwd)
        # x_h_bwd = layers.Dropout(self.drop_prob)(x_h_bwd)

        # x_a_fwd, w_a_fwd = BahdanauAttention(1024)(x_h_fwd, x_e)
        # x_a_bwd, w_a_bwd = BahdanauAttention(1024)(x_h_bwd, x_e)

        # x = tf.concat([x_h_fwd, x_a_fwd, x_h_bwd, x_a_bwd], axis=-1)

        # x = layers.Dense(1, activation='sigmoid', name='prediction')(x)
        # x = tf.math.add(tf.math.multiply(x, 240.0), 40.0)

        return tf.keras.Model(inputs=x0, outputs=x)


        # w_a0_fwd = tf.argsort(w_a0_fwd, direction="DESCENDING")[:,:5]
        # w_a0_bwd = tf.argsort(w_a0_bwd, direction="DESCENDING")[:,:5]
        # w_a1_fwd = tf.argsort(w_a1_fwd, direction="DESCENDING")[:,:5]
        # w_a1_bwd = tf.argsort(w_a1_bwd, direction="DESCENDING")[:,:5]
        # w_a2_fwd = tf.argsort(w_a2_fwd, direction="DESCENDING")[:,:5]
        # w_a2_bwd = tf.argsort(w_a2_bwd, direction="DESCENDING")[:,:5]
        # w_a3_fwd = tf.argsort(w_a3_fwd, direction="DESCENDING")[:,:5]
        # w_a3_bwd = tf.argsort(w_a3_bwd, direction="DESCENDING")[:,:5]
        # w_a4_fwd = tf.argsort(w_a4_fwd, direction="DESCENDING")[:,:5]
        # w_a4_bwd = tf.argsort(w_a4_bwd, direction="DESCENDING")[:,:5]
        # w_a5_fwd = tf.argsort(w_a5_fwd, direction="DESCENDING")[:,:5]
        # w_a5_bwd = tf.argsort(w_a5_bwd, direction="DESCENDING")[:,:5]
        # return tf.keras.Model(inputs=x0, outputs=[x, w_a0_fwd, w_a0_bwd, w_a1_fwd, w_a1_bwd, w_a2_fwd, w_a2_bwd, w_a3_fwd, w_a3_bwd, w_a4_fwd, w_a4_bwd, w_a5_fwd, w_a5_bwd])
        


    @tf.function(experimental_relax_shapes=True)
    def train_step(self, train_tuple):
        
        images = train_tuple[0]
        labels = train_tuple[self.enumerate_index]
        sample_weight = None

        if self.class_weights_index != -1:
            sample_weight = train_tuple[self.class_weights_index]

        with tf.GradientTape() as tape:
            
            x_c = self.gru_class(images, training=True)
            loss = self.loss(labels, x_c, sample_weight=sample_weight)

            var_list = self.trainable_variables

            gradients = tape.gradient(loss, var_list)
            self.optimizer.apply_gradients(zip(gradients, var_list))

            return loss, x_c

    def valid_step(self, dataset_validation):

        loss = 0
        for valid_tuple in dataset_validation:
            images = valid_tuple[0]
            labels = valid_tuple[self.enumerate_index]
            
            x_c = self.gru_class(images, training=False)

            loss += self.validation_loss(labels, x_c)

            self.validation_metric.update_state(labels, x_c)

        metric = self.validation_metric.result()

        tf.summary.scalar('validation_loss', loss, step=self.global_validation_step)
        tf.summary.scalar('validation_acc', metric, step=self.global_validation_step)
        self.global_validation_step += 1

        print("val loss", loss.numpy(), "mae", metric.numpy())
        
        improved = False
        if loss < self.global_validation_metric:
            self.global_validation_metric = loss
            improved = True

        return improved

    def get_checkpoint_manager(self):
        return tf.train.Checkpoint(
            gru_class=self.gru_class,
            optimizer=self.optimizer)

    def summary(self, train_tuple, tr_step, step):
        
        sample_weight = None
        if self.class_weights_index != -1:
            sample_weight = train_tuple[self.class_weights_index]

        labels = tf.reshape(train_tuple[1], [-1])

        loss = tr_step[0]
        prediction = tf.reshape(tr_step[1], [-1])

        self.metrics_train.update_state(labels, prediction, sample_weight=sample_weight)
        metrics_result = self.metrics_train.result()

        print("step", step, "loss", loss.numpy(), "mae", metrics_result.numpy())
        print(labels.numpy())
        print(prediction.numpy())

        # x_spc = train_tuple[0][:,0,:,512:513]
        # print(x_spc.numpy())
        
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('loss', loss, step=step)
        tf.summary.scalar('mae', metrics_result, step=step)

    def save_model(self, save_model):
        self.gru_class.summary()
        self.gru_class.save(save_model)





# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras import layers
# import json
# import os
# import glob
# import sys

# class Attention(layers.Layer):
#     def __init__(self, units, k=5):
#         super(Attention, self).__init__()

#         self.W1 = tf.keras.layers.Dense(units)
#         self.W2 = tf.keras.layers.Dense(units)
#         self.V = tf.keras.layers.Dense(1)
#         self.k = k

#     def call(self, query, values):

#         # query hidden state shape == (batch_size, hidden size)
#         # query_with_time_axis shape == (batch_size, 1, hidden size)
#         # values shape == (batch_size, max_len, hidden size)
#         # we are doing this to broadcast addition along the time axis to calculate the score
#         query_with_time_axis = tf.expand_dims(query, 1)

#         # score shape == (batch_size, max_length, 1)
#         # we get 1 at the last axis because we are applying score to self.V
#         # the shape of the tensor before applying self.V is (batch_size, max_length, units)
#         score = self.V(tf.nn.tanh(
#             self.W1(query_with_time_axis) + self.W2(values)))

#         # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
#         # min_score = tf.reshape(min_score, [-1, 1, 1])
#         # score_mask = tf.greater_equal(score, min_score)
#         # score_mask = tf.cast(score_mask, tf.float32)
#         # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

#         # attention_weights shape == (batch_size, max_length, 1)
#         # attention_weights = tf.nn.softmax(score, axis=1)

#         score = tf.reshape(score, [-1, tf.shape(score)[1]])
#         min_score = tf.reduce_min(tf.math.top_k(score, k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)

#         score_mask = tf.greater_equal(score, min_score)
#         score_mask = tf.cast(score_mask, tf.float32)
        
#         attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

#         # context_vector shape after sum == (batch_size, hidden_size)
#         context_vector = tf.reshape(attention_weights, [-1, tf.shape(attention_weights)[1], 1]) * values
#         context_vector = tf.reduce_sum(context_vector, axis=1)

#         # Pick k elements from the tensor

#         attention_weights_i = tf.argsort(attention_weights, direction='DESCENDING')
#         values = tf.gather(values, attention_weights_i[:,:self.k], axis=1, batch_dims=1)

#         return context_vector, attention_weights, values


# class BahdanauAttention(tf.keras.layers.Layer):
#   def __init__(self, units):
#     super(BahdanauAttention, self).__init__()
#     self.W1 = tf.keras.layers.Dense(units, use_bias=False)
#     self.W2 = tf.keras.layers.Dense(units, use_bias=False)
#     self.V = tf.keras.layers.Dense(1, use_bias=False)
#     # self.k = k

#   def call(self, query, values):
#     # query hidden state shape == (batch_size, hidden size)
#     # query_with_time_axis shape == (batch_size, 1, hidden size)
#     # values shape == (batch_size, max_len, hidden size)
#     # we are doing this to broadcast addition along the time axis to calculate the score
#     query_with_time_axis = tf.expand_dims(query, 1)

#     # score shape == (batch_size, max_length, 1)
#     # we get 1 at the last axis because we are applying score to self.V
#     # the shape of the tensor before applying self.V is (batch_size, max_length, units)
#     score = self.V(tf.nn.tanh(
#         self.W1(query_with_time_axis) + self.W2(values)))

#     # min_score = tf.reduce_min(tf.math.top_k(tf.reshape(score, [-1, tf.shape(score)[1]]), k=self.k, sorted=False, name=None)[0], axis=1, keepdims=True)
#     # min_score = tf.reshape(min_score, [-1, 1, 1])
#     # score_mask = tf.greater_equal(score, min_score)
#     # score_mask = tf.cast(score_mask, tf.float32)
#     # attention_weights = tf.multiply(tf.exp(score), score_mask) / tf.reduce_sum(tf.multiply(tf.exp(score), score_mask), axis=1, keepdims=True)

#     # attention_weights shape == (batch_size, max_length, 1)
#     attention_weights = tf.nn.softmax(score, axis=1)

#     # context_vector shape after sum == (batch_size, hidden_size)
#     context_vector = attention_weights * values
#     context_vector = tf.reduce_sum(context_vector, axis=1)

#     return context_vector, attention_weights

# class SigmoidCrossEntropy(tf.keras.losses.Loss):
#     def __init__(self, max_value, reduction=tf.keras.losses.Reduction.AUTO):
#         super(SigmoidCrossEntropy, self).__init__()
#         self.reduction = reduction
#         self.max_value = max_value

#     def call(self, y_true, y_pred, sample_weight=None):

#         y_true = tf.math.divide(y_true, self.max_value)
#         loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
#         if sample_weight is not None:
#             loss = tf.multiply(loss, sample_weight)

#         if self.reduction == 'sum':
#             return tf.reduce_sum(loss)
#         return tf.reduce_mean(loss)

# class GruAtt(tf.keras.layers.Layer):
#     def __init__(self, gru_units=1024, attention_units=1024, drop_prob=0):
#         super(GruAtt, self).__init__()
        
#         self.drop_prob = drop_prob

#         self.bi_gru = layers.Bidirectional(layers.GRU(units=gru_units, activation='tanh', dropout=self.drop_prob, return_sequences=True, return_state=True))
#         self.drop0 = layers.Dropout(self.drop_prob)
#         self.drop1 = layers.Dropout(self.drop_prob)
#         self.drop2 = layers.Dropout(self.drop_prob)
#         self.att_fwd = BahdanauAttention(attention_units)
#         self.att_bwd = BahdanauAttention(attention_units)

#         self.concat = layers.Concatenate()

#     def call(self, x):
#         x_e, x_h_fwd, x_h_bwd = self.bi_gru(x)
#         x_e = self.drop0(x_e)
#         x_h_fwd = self.drop1(x_h_fwd)
#         x_h_bwd = self.drop2(x_h_bwd)

#         x_a_fwd, w_a_fwd = self.att_fwd(x_h_fwd, x_e)
#         x_a_bwd, w_a_bwd = self.att_bwd(x_h_bwd, x_e)

#         x = self.concat([x_h_fwd, x_a_fwd, x_h_bwd, x_a_bwd])

#         return x, w_a_fwd, w_a_bwd


# class NN(tf.keras.Model):

#     def __init__(self, tf_inputs, args):
#         super(NN, self).__init__()
        
#         learning_rate = args.learning_rate
#         decay_steps = args.decay_steps
#         decay_rate = args.decay_rate
#         staircase = args.staircase
#         drop_prob = args.drop_prob

#         data_description = tf_inputs.get_data_description()
#         self.num_channels = data_description[data_description["data_keys"][0]]["shape"][-1]

#         self.num_classes = 2
#         self.class_weights_index = -1
#         self.enumerate_index = 1

#         if "enumerate" in data_description:
#             self.enumerate_index = data_description["data_keys"].index(data_description["enumerate"])

#             if(data_description[data_description["data_keys"][self.enumerate_index]]["num_class"]):
#                 self.num_classes = data_description[data_description["data_keys"][self.enumerate_index]]["num_class"]
#                 print("Number of classes in data description", self.num_classes)
#                 if "class_weights" in data_description["data_keys"]:
#                     self.class_weights_index = data_description["data_keys"].index("class_weights")
#                     print("Using weights index", self.class_weights_index)

#         self.drop_prob = drop_prob

#         self.gru_class = self.make_gru_network()
#         self.gru_class.summary()
        
#         self.max_value = 290.0
#         # self.loss = SigmoidCrossEntropy(self.max_value)
#         self.loss = tf.keras.losses.MeanSquaredError()
#         # self.loss = tf.keras.losses.Huber(delta=5.0, reduction=tf.keras.losses.Reduction.SUM)
#         # self.loss = tf.keras.losses.LogCosh()
#         self.metrics_train = tf.keras.metrics.MeanAbsoluteError()

#         if decay_rate != 0.0:
#             lr = tf.keras.optimizers.schedules.ExponentialDecay(learning_rate, decay_steps, decay_rate, staircase)
#         else:
#             lr = learning_rate

#         self.optimizer = tf.keras.optimizers.Adam(lr)
        

#         self.validation_loss = tf.keras.losses.MeanSquaredError()
#         # self.validation_loss = tf.keras.losses.Huber(delta=5.0, reduction=tf.keras.losses.Reduction.SUM)
#         # self.validation_loss = tf.keras.losses.MeanSquaredError()
#         # self.validation_loss = SigmoidCrossEntropy(self.max_value)
#         self.validation_metric = tf.keras.metrics.MeanAbsoluteError()

#         self.global_validation_metric = float("inf")
#         self.global_validation_step = args.in_epoch

#     def make_gru_network(self):

        
#         x0 = tf.keras.Input(shape=[None, 6, self.num_channels])
#         x = layers.Masking(mask_value=-1.0)(x0)

#         x = layers.BatchNormalization()(x)

#         x_0 = layers.Reshape([-1, self.num_channels])(x[:,:,0,:])
#         x_1 = layers.Reshape([-1, self.num_channels])(x[:,:,1,:])
#         x_2 = layers.Reshape([-1, self.num_channels])(x[:,:,2,:])
#         x_3 = layers.Reshape([-1, self.num_channels])(x[:,:,3,:])
#         x_4 = layers.Reshape([-1, self.num_channels])(x[:,:,4,:])
#         x_5 = layers.Reshape([-1, self.num_channels])(x[:,:,5,:])

#         x_0, w_a0_fwd, w_a0_bwd = GruAtt(gru_units=1024, attention_units=2048, drop_prob=self.drop_prob)(x_0)
#         x_1, w_a1_fwd, w_a1_bwd = GruAtt(gru_units=1024, attention_units=2048, drop_prob=self.drop_prob)(x_1)
#         x_2, w_a2_fwd, w_a2_bwd = GruAtt(gru_units=1024, attention_units=2048, drop_prob=self.drop_prob)(x_2)
#         x_3, w_a3_fwd, w_a3_bwd = GruAtt(gru_units=1024, attention_units=2048, drop_prob=self.drop_prob)(x_3)
#         x_4, w_a4_fwd, w_a4_bwd = GruAtt(gru_units=1024, attention_units=2048, drop_prob=self.drop_prob)(x_4)
#         x_5, w_a5_fwd, w_a5_bwd = GruAtt(gru_units=1024, attention_units=2048, drop_prob=self.drop_prob)(x_5)

#         x_0 = tf.expand_dims(x_0, axis=1)
#         x_1 = tf.expand_dims(x_1, axis=1)
#         x_2 = tf.expand_dims(x_2, axis=1)
#         x_3 = tf.expand_dims(x_3, axis=1)
#         x_4 = tf.expand_dims(x_4, axis=1)
#         x_5 = tf.expand_dims(x_5, axis=1)

#         x = layers.Concatenate(axis=1)([x_0, x_1, x_2, x_3, x_4, x_5])

#         x, w_a_fwd, w_a_bwd = GruAtt(gru_units=2048, attention_units=4096, drop_prob=self.drop_prob)(x)

#         # x_e0, x_h_fwd0, x_h_bwd0 = layers.Bidirectional(layers.GRU(units=512, activation='tanh', dropout=self.drop_prob, return_sequences=True, return_state=True))(x_0)
#         # x_e0 = layers.Dropout(self.drop_prob)(x_e0)
#         # x_h_fwd0 = layers.Dropout(self.drop_prob)(x_h_fwd0)
#         # x_h_bwd0 = layers.Dropout(self.drop_prob)(x_h_bwd0)
#         # x_a_fwd0, w_a_fwd0 = BahdanauAttention(1024)(x_h_fwd0, x_e0)
#         # x_a_bwd0, w_a_bwd0 = BahdanauAttention(1024)(x_h_bwd0, x_e0)

#         # x_e1, x_h_fwd1, x_h_bwd1 = layers.Bidirectional(layers.GRU(units=512, activation='tanh', dropout=self.drop_prob, return_sequences=True, return_state=True))(x_1)
#         # x_e1 = layers.Dropout(self.drop_prob)(x_e1)
#         # x_h_fwd1 = layers.Dropout(self.drop_prob)(x_h_fwd1)
#         # x_h_bwd1 = layers.Dropout(self.drop_prob)(x_h_bwd1)
#         # x_a_fwd1, w_a_fwd1 = BahdanauAttention(1024)(x_h_fwd1, x_e1)
#         # x_a_bwd1, w_a_bwd1 = BahdanauAttention(1024)(x_h_bwd1, x_e1)

#         # x_e2, x_h_fwd2, x_h_bwd2 = layers.Bidirectional(layers.GRU(units=512, activation='tanh', dropout=self.drop_prob, return_sequences=True, return_state=True))(x_2)
#         # x_e2 = layers.Dropout(self.drop_prob)(x_e2)
#         # x_h_fwd2 = layers.Dropout(self.drop_prob)(x_h_fwd2)
#         # x_h_bwd2 = layers.Dropout(self.drop_prob)(x_h_bwd2)
#         # x_a_fwd2, w_a_fwd2 = BahdanauAttention(1024)(x_h_fwd2, x_e2)
#         # x_a_bwd2, w_a_bwd2 = BahdanauAttention(1024)(x_h_bwd2, x_e2)

#         # x = tf.concat([x_h_fwd0, x_a_fwd0, x_h_bwd0, x_a_bwd0, x_h_fwd1, x_a_fwd1, x_h_bwd1, x_a_bwd1, x_h_fwd2, x_a_fwd2, x_h_bwd2, x_a_bwd2], axis=-1)
#         x = layers.Dense(1, activation='sigmoid', name='prediction', use_bias=False)(x)
#         # x = tf.math.add(tf.math.multiply(x, 240.0), 40.0)
#         x = tf.math.add(tf.math.multiply(x, 90.0), 190.0)

  
#         # x_e, x_h_fwd, x_h_bwd = layers.Bidirectional(layers.GRU(units=512, activation='tanh', use_bias=False, kernel_initializer="glorot_normal", dropout=self.drop_prob, return_sequences=True, return_state=True), name="bi_gru0")(x)
#         # x_e = layers.Dropout(self.drop_prob)(x_e)
#         # x_h_fwd = layers.Dropout(self.drop_prob)(x_h_fwd)
#         # x_h_bwd = layers.Dropout(self.drop_prob)(x_h_bwd)

#         # x_a_fwd, w_a_fwd = BahdanauAttention(1024)(x_h_fwd, x_e)
#         # x_a_bwd, w_a_bwd = BahdanauAttention(1024)(x_h_bwd, x_e)

#         # x = tf.concat([x_h_fwd, x_a_fwd, x_h_bwd, x_a_bwd], axis=-1)

#         # x = layers.Dense(1, activation='sigmoid', name='prediction')(x)
#         # x = tf.math.add(tf.math.multiply(x, 240.0), 40.0)

#         return tf.keras.Model(inputs=x0, outputs=x)


#         # w_a0_fwd = tf.argsort(w_a0_fwd, direction="DESCENDING")[:,:5]
#         # w_a0_bwd = tf.argsort(w_a0_bwd, direction="DESCENDING")[:,:5]
#         # w_a1_fwd = tf.argsort(w_a1_fwd, direction="DESCENDING")[:,:5]
#         # w_a1_bwd = tf.argsort(w_a1_bwd, direction="DESCENDING")[:,:5]
#         # w_a2_fwd = tf.argsort(w_a2_fwd, direction="DESCENDING")[:,:5]
#         # w_a2_bwd = tf.argsort(w_a2_bwd, direction="DESCENDING")[:,:5]
#         # w_a3_fwd = tf.argsort(w_a3_fwd, direction="DESCENDING")[:,:5]
#         # w_a3_bwd = tf.argsort(w_a3_bwd, direction="DESCENDING")[:,:5]
#         # w_a4_fwd = tf.argsort(w_a4_fwd, direction="DESCENDING")[:,:5]
#         # w_a4_bwd = tf.argsort(w_a4_bwd, direction="DESCENDING")[:,:5]
#         # w_a5_fwd = tf.argsort(w_a5_fwd, direction="DESCENDING")[:,:5]
#         # w_a5_bwd = tf.argsort(w_a5_bwd, direction="DESCENDING")[:,:5]
#         # return tf.keras.Model(inputs=x0, outputs=[x, w_a0_fwd, w_a0_bwd, w_a1_fwd, w_a1_bwd, w_a2_fwd, w_a2_bwd, w_a3_fwd, w_a3_bwd, w_a4_fwd, w_a4_bwd, w_a5_fwd, w_a5_bwd])
        


#     @tf.function(experimental_relax_shapes=True)
#     def train_step(self, train_tuple):
        
#         images = train_tuple[0]
#         labels = train_tuple[self.enumerate_index]
#         sample_weight = None

#         if self.class_weights_index != -1:
#             sample_weight = train_tuple[self.class_weights_index]

#         with tf.GradientTape() as tape:
            
#             x_c = self.gru_class(images, training=True)
#             loss = self.loss(labels, x_c, sample_weight=sample_weight)

#             var_list = self.trainable_variables

#             gradients = tape.gradient(loss, var_list)
#             self.optimizer.apply_gradients(zip(gradients, var_list))

#             return loss, x_c

#     def valid_step(self, dataset_validation):

#         loss = 0
#         for valid_tuple in dataset_validation:
#             images = valid_tuple[0]
#             labels = valid_tuple[self.enumerate_index]
            
#             x_c = self.gru_class(images, training=False)

#             loss += self.validation_loss(labels, x_c)

#             self.validation_metric.update_state(labels, x_c)

#         metric = self.validation_metric.result()

#         tf.summary.scalar('validation_loss', loss, step=self.global_validation_step)
#         tf.summary.scalar('validation_acc', metric, step=self.global_validation_step)
#         self.global_validation_step += 1

#         print("val loss", loss.numpy(), "mae", metric.numpy())
        
#         improved = False
#         if loss < self.global_validation_metric:
#             self.global_validation_metric = loss
#             improved = True

#         return improved

#     def get_checkpoint_manager(self):
#         return tf.train.Checkpoint(
#             gru_class=self.gru_class,
#             optimizer=self.optimizer)

#     def summary(self, train_tuple, tr_step, step):
        
#         sample_weight = None
#         if self.class_weights_index != -1:
#             sample_weight = train_tuple[self.class_weights_index]

#         labels = tf.reshape(train_tuple[1], [-1])

#         loss = tr_step[0]
#         prediction = tf.reshape(tr_step[1], [-1])

#         self.metrics_train.update_state(labels, prediction, sample_weight=sample_weight)
#         metrics_result = self.metrics_train.result()

#         print("step", step, "loss", loss.numpy(), "mae", metrics_result.numpy())
#         print(labels.numpy())
#         print(prediction.numpy())
        
#         tf.summary.scalar('loss', loss, step=step)
#         tf.summary.scalar('loss', loss, step=step)
#         tf.summary.scalar('mae', metrics_result, step=step)

#     def save_model(self, save_model):
#         self.gru_class.summary()
#         self.gru_class.save(save_model)