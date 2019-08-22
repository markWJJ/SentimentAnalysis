# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
'''

from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def focal_loss(prediction_tensor, target_tensor, weights=None, alpha=0.5, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    print(sigmoid_p.get_shape())
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    pos_p_sub = array_ops.where(target_tensor >= sigmoid_p, target_tensor - sigmoid_p, zeros)
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_mean(per_entry_cross_ent)

def embedding(inputs,
              vocab_size,
              num_units,
              zero_pad=True,
              scale=True,
              scope="embedding",
              reuse=None):
    '''Embeds a given tensor.

    Args:
      inputs: A `Tensor` with type `int32` or `int64` containing the ids
         to be looked up in `lookup table`.
      vocab_size: An int. Vocabulary size.
      num_units: An int. Number of embedding hidden units.
      zero_pad: A boolean. If True, all the values of the fist row (id 0)
        should be constant zeros.
      scale: A boolean. If True. the outputs is multiplied by sqrt num_units.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A `Tensor` with one more rank than inputs's. The last dimensionality
        should be `num_units`.

    For example,

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[ 0.          0.        ]
      [ 0.09754146  0.67385566]
      [ 0.37864095 -0.35689294]]

     [[-1.01329422 -1.09939694]
      [ 0.7521342   0.38203377]
      [-0.04973143 -0.06210355]]]
    ```

    ```
    import tensorflow as tf

    inputs = tf.to_int32(tf.reshape(tf.range(2*3), (2, 3)))
    outputs = embedding(inputs, 6, 2, zero_pad=False)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print sess.run(outputs)
    >>
    [[[-0.19172323 -0.39159766]
      [-0.43212751 -0.66207761]
      [ 1.03452027 -0.26704335]]

     [[-0.11634696 -0.35983452]
      [ 0.50208133  0.53509563]
      [ 1.22204471 -0.96587461]]]    
    ```    
    '''
    with tf.variable_scope(scope, reuse=reuse):
        lookup_table = tf.get_variable('lookup_table',
                                       dtype=tf.float32,
                                       shape=[vocab_size, num_units],
                                       trainable=False,
                                       initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, inputs)

        if scale:
            outputs = outputs * (num_units ** 0.5)
            # ?
    return outputs

# def self_attention(lstm_outs,sent_mask):
#     '''
#     attention
#     :param lstm_outs:
#     :param sent_mask:
#     :return:
#     '''
#     with tf.variable_scope(name_or_scope='attention'):
#         if isinstance(lstm_outs,list):
#             lstm_outs=tf.stack(lstm_outs,1)
#         hidden = lstm_outs.get_shape().as_list()[-1]
#         V=tf.Variable(tf.random_uniform(shape=(hidden,1),dtype=tf.float32))
#         logit=tf.layers.dense(lstm_outs,hidden,activation=tf.nn.tanh,use_bias=True)
#         logit=tf.einsum('ijk,kl->ijl',logit,V)
#         logit=tf.squeeze(logit,-1)
#         logit=tf.multiply(logit,tf.cast(sent_mask,tf.float32))
#         soft_logit=tf.nn.softmax(logit,1)
#         soft_logit=tf.expand_dims(soft_logit,-1)
#         attention_out=tf.einsum('ijk,ijl->ilk',lstm_outs,soft_logit)
#         attention_out=tf.squeeze(attention_out,1)
#         return attention_out

def exp_mask(logits, mask, mask_is_length=True):
    """Exponential mask for logits.

      Logits cannot be masked with 0 (i.e. multiplying boolean mask)
      because expnentiating 0 becomes 1. `exp_mask` adds very large negative value
      to `False` portion of `mask` so that the portion is effectively ignored
      when exponentiated, e.g. softmaxed.

    Args:
        logits: Arbitrary-rank logits tensor to be masked.
        mask: `boolean` type mask tensor.
          Could be same shape as logits (`mask_is_length=False`)
          or could be length tensor of the logits (`mask_is_length=True`).
        mask_is_length: `bool` value. whether `mask` is boolean mask.
    Returns:
        Masked logits with the same shape of `logits`.
    """
    if mask_is_length:
        mask = tf.sequence_mask(mask, maxlen=tf.shape(logits)[-1])
    return logits + (1.0 - tf.cast(mask, 'float')) * -1e12

def self_attention(tensor,
             mask=None,
             mask_is_length=True,
             logit_fn=None,
             scale_dot=False,
             normalizer=tf.nn.softmax,
             scope=None,
             reuse=False):
    """Performs self attention.

        Performs self attention to obtain single vector representation for a sequence of vectors.

    Args:
        tensor: [batch_size, sequence_length, hidden_size]-shaped tensor
        mask: Length mask (shape of [batch_size]) or boolean mask ([batch_size, sequence_length])
        mask_is_length: `True` if `mask` is length mask, `False` if it is boolean mask
        logit_fn: `logit_fn(tensor)` to obtain logits.
        scale_dot: `bool`, whether to scale the dot product by dividing by sqrt(hidden_size).
        normalizer: function to normalize logits.
        scope: `string` for defining variable scope
        reuse: Reuse if `True`.
    Returns:
        [batch_size, hidden_size]-shaped tensor.
    """
    assert len(tensor.get_shape()) == 3, 'The rank of `tensor` must be 3 but got {}.'.format(
        len(tensor.get_shape()))
    with tf.variable_scope(scope or 'self_att', reuse=reuse):
        hidden_size = tensor.get_shape().as_list()[-1]
        if logit_fn is None:
            logits = tf.layers.dense(tensor, hidden_size, activation=tf.tanh)
            logits = tf.squeeze(tf.layers.dense(logits, 1), 2)
        else:
            logits = logit_fn(tensor)
        if scale_dot:
            logits /= tf.sqrt(hidden_size)
        if mask is not None:
            logits = exp_mask(logits, mask, mask_is_length=mask_is_length)
        weights = normalizer(logits)
        out = tf.reduce_sum(tf.expand_dims(weights, -1) * tensor, 1)
        return out,weights


def self_attention_topk(tensor,
             mask=None,
             mask_is_length=True,
             logit_fn=None,
             scale_dot=False,
             normalizer=tf.nn.softmax,
             scope=None,
             reuse=False,
            top=None):
    """Performs self attention.

        Performs self attention to obtain single vector representation for a sequence of vectors.

    Args:
        tensor: [batch_size, sequence_length, hidden_size]-shaped tensor
        mask: Length mask (shape of [batch_size]) or boolean mask ([batch_size, sequence_length])
        mask_is_length: `True` if `mask` is length mask, `False` if it is boolean mask
        logit_fn: `logit_fn(tensor)` to obtain logits.
        scale_dot: `bool`, whether to scale the dot product by dividing by sqrt(hidden_size).
        normalizer: function to normalize logits.
        scope: `string` for defining variable scope
        reuse: Reuse if `True`.
    Returns:
        [batch_size, hidden_size]-shaped tensor.
    """
    assert len(tensor.get_shape()) == 3, 'The rank of `tensor` must be 3 but got {}.'.format(
        len(tensor.get_shape()))
    with tf.variable_scope(scope or 'self_att', reuse=reuse):
        hidden_size = tensor.get_shape().as_list()[-1]
        if logit_fn is None:
            logits =tf.layers.dense(tensor, hidden_size, activation=tf.tanh)
            logits = tf.squeeze(tf.layers.dense(logits, 1), 2)
            # logits = tf.sign(logits)
        else:
            logits = logit_fn(tensor)
        if scale_dot:
            logits /= tf.sqrt(hidden_size)
        if mask is not None:
            logits = self.exp_mask(logits, mask, mask_is_length=mask_is_length)
        weights = normalizer(logits)

        # weights=tf.nn.tanh(weights)
        # weights = self.exp_mask(weights, mask, mask_is_length=mask_is_length)
        #
        # weights=tf.nn.softmax(weights,-1)
        batch_size=tf.shape(logits)[0]
        max_length=tensor.get_shape().as_list()[1]
        out_dim=int(tensor.get_shape()[-1])
        sort_res=tf.nn.top_k(weights,top,sorted=True)
        sort_res_vaule=sort_res[0]
        out_sort_res_vaule=tf.nn.softmax(sort_res_vaule)
        sort_res_index=sort_res[1]
        sort_res_vaule_ls=tf.unstack(sort_res_vaule,top,1)
        sort_res_index_ls=tf.unstack(sort_res_index,top,1)
        res=[]
        out_sent=[]
        for res_v,res_i in zip(sort_res_vaule_ls,sort_res_index_ls):
            index = tf.cast(tf.range(0, batch_size) * max_length,tf.int32) + (res_i)
            s = tf.reshape(tensor, (-1, out_dim))
            ss = tf.gather(s, index)
            out_sent.append(ss)
            ss=tf.multiply(ss,tf.expand_dims(res_v, -1))
            res.append(ss)
        res=tf.stack(res,1)
        out_sent=tf.stack(out_sent,1)
        out = tf.reduce_sum(res, 1)
        return out,out_sort_res_vaule,sort_res_index,out_sent

def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    '''Sinusoidal Positional_Encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
        A 'Tensor' with one more rank than inputs's, with the dimensionality should be 'num_units'
    '''

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, 2. * i / num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(position_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, position_ind)

        if scale:
            outputs = outputs * num_units ** 0.5

        return outputs

        # ?


def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0.1,
                        is_training=True,
                        causality=False,
                        scope="multihead_attention",
                        reuse=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      dropout_rate: A floating point number.
      is_training: Boolean. Controller of mechanism for dropout.
      causality: Boolean. If true, units that reference the future are masked. 
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns
      A 3d tensor with shape of (N, T_q, C)  
    '''

    # 其中x，y的shape为[N,T]，N即batch_size的大小，T为最大句子长度maxlen，默认为10
    #
    #
    #
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        # Linear projections
        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)

        # Multiplication
        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # (h*N, T_q, T_k)

        # Scale
        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)

        # Key Masking
        key_masks = tf.sign(tf.abs(tf.reduce_sum(keys, axis=-1)))  # (N, T_k)
        key_masks = tf.tile(key_masks, [num_heads, 1])  # (h*N, T_k)
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])  # (h*N, T_q, T_k)

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Causality = Future blinding
        if causality:
            diag_vals = tf.ones_like(outputs[0, :, :])  # (T_q, T_k)
            tril = tf.contrib.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  # (T_q, T_k)
            masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])  # (h*N, T_q, T_k)

            paddings = tf.ones_like(masks) * (-2 ** 32 + 1)
            outputs = tf.where(tf.equal(masks, 0), paddings, outputs)  # (h*N, T_q, T_k)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.sign(tf.abs(tf.reduce_sum(queries, axis=-1)))  # (N, T_q)
        query_masks = tf.tile(query_masks, [num_heads, 1])  # (h*N, T_q)
        query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])  # (h*N, T_q, T_k)
        outputs *= query_masks  # broadcasting. (N, T_q, C)

        # Dropouts
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

        # Weighted sum
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = normalize(outputs)  # (N, T_q, C)

    return outputs


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    '''Point-wise feed forward net.

    Args:
      inputs: A 3d tensor with shape of [N, T, C].
      num_units: A list of two integers.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)

    return outputs


def label_smoothing(inputs, epsilon=0.1):
    '''Applies label smoothing. See https://arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of [N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

    For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    '''
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1 - epsilon) * inputs) + (epsilon / K)


def beam_search(inputs, beam_width):
    test_preds = []
    for i in range(inputs.get_shape().as_list()[0]):
        path = [[] for _ in range(beam_width)]
        probs = [[] for _ in range(beam_width)]
        log_prob_a = tf.nn.log_softmax(inputs[i][0])
        top_k_prob, top_k_indexs = tf.nn.top_k(log_prob_a, beam_width)

        for k in range(beam_width):
            path[k].extend([top_k_indexs[k]])
            probs[k].extend([top_k_prob[k]])

        for j in range(1, inputs.get_shape().as_list()[0]):
            print(j)

            tmp = []
            tmp_idx = []

            log_prob = tf.nn.log_softmax(inputs[j])

            # print(sess.run(log_prob))

            for k in range(beam_width):
                log_prob_a = tf.add(log_prob, probs[k][-1])
                # print(sess.run(log_prob))
                current_path = path[k]
                # print(sess.run(current_path))
                top_k_prob, top_k_indexs = tf.nn.top_k(log_prob, beam_width)
                # print(sess.run(top_k_indexs))
                # print(sess.run(top_k_indexs[0]))
                for i in range(beam_width):
                    new_path = tf.concat([current_path, [top_k_indexs[i]]], 0)
                    tmp_idx.append(new_path)

                tmp = tf.concat([tmp, top_k_prob], 0)

                # print(sess.run(tmp_idx[1]))
                # if j >=10:
                a_log_prob, a_indexs = tf.nn.top_k(tmp, beam_width)
                for k in range(beam_width):
                    path[k] = tmp_idx[sess.run(a_indexs[k])]
                    probs[k] = a_log_prob

            test_preds.append(path)
    return test_preds


def mean_pool(input_tensor, sequence_length=None):
    """
    Given an input tensor (e.g., the outputs of a LSTM), do mean pooling
    over the last dimension of the input.

    For example, if the input was the output of a LSTM of shape
    (batch_size, sequence length, hidden_dim), this would
    calculate a mean pooling over the last dimension (taking the padding
    into account, if provided) to output a tensor of shape
    (batch_size, hidden_dim).

    Parameters
    ----------
    input_tensor: Tensor
        An input tensor, preferably the output of a tensorflow RNN.
        The mean-pooled representation of this output will be calculated
        over the last dimension.

    sequence_length: Tensor, optional (default=None)
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    mean_pooled_output: Tensor
        A tensor of one less dimension than the input, with the size of the
        last dimension equal to the hidden dimension state size.
    """
    with tf.name_scope("mean_pool"):
        # shape (batch_size, sequence_length)
        input_tensor_sum = tf.reduce_sum(input_tensor, axis=-2)

        # If sequence_length is None, divide by the sequence length
        # as indicated by the input tensor.
        if sequence_length is None:
            sequence_length = tf.shape(input_tensor)[-2]

        # Expand sequence length from shape (batch_size,) to
        # (batch_size, 1) for broadcasting to work.
        expanded_sequence_length = tf.cast(tf.expand_dims(sequence_length, -1),
                                           "float32") + 1e-08

        # Now, divide by the length of each sequence.
        # shape (batch_size, sequence_length)
        mean_pooled_input = (input_tensor_sum /
                             expanded_sequence_length)
        return mean_pooled_input

def sent_encoder(sent_word_emb,hidden_dim,sequence_length,name,dropout=0.0,reuse=False):
    '''
    句编码
    :param sent_word_emb:
    :param hidden_dim:
    :param name:
    :return:
    '''
    with tf.variable_scope(name_or_scope=name,reuse=reuse):
        lstm_cell=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_cell_1=tf.contrib.rnn.BasicLSTMCell(hidden_dim)
        lstm_cell=tf.nn.rnn_cell.DropoutWrapper(lstm_cell, output_keep_prob=1.0-dropout)
        lstm_cell_1=tf.nn.rnn_cell.DropoutWrapper(lstm_cell_1, output_keep_prob=1.0-dropout)
        encoder,_=tf.nn.bidirectional_dynamic_rnn(
            lstm_cell,
            lstm_cell_1,
            sent_word_emb,
            dtype=tf.float32,
            sequence_length=sequence_length, )
        # encoder,_=tf.nn.static_rnn(lstm_cell,sent_word_embs,sequence_length=sequence_length,dtype=tf.float32)
        encoder=tf.concat(encoder,2)
        return encoder


def last_relevant_output(output, sequence_length):
    """
    Given the outputs of a LSTM, get the last relevant output that
    is not padding. We assume that the last 2 dimensions of the input
    represent (sequence_length, hidden_size).

    Parameters
    ----------
    output: Tensor
        A tensor, generally the output of a tensorflow RNN.
        The tensor index sequence_lengths+1 is selected for each
        instance in the output.

    sequence_length: Tensor
        A tensor of dimension (batch_size, ) indicating the length
        of the sequences before padding was applied.

    Returns
    -------
    last_relevant_output: Tensor
        The last relevant output (last element of the sequence), as retrieved
        by the output Tensor and indicated by the sequence_length Tensor.
    """
    with tf.name_scope("last_relevant_output"):
        batch_size = tf.shape(output)[0]
        max_length = tf.shape(output)[-2]
        out_size = int(output.get_shape()[-1])
        index = tf.range(0, batch_size) * max_length + (sequence_length - 1)
        flat = tf.reshape(output, [-1, out_size])
        relevant = tf.gather(flat, index)
        return relevant


def get_shape_list(tensor, expected_rank=None, name=None):
  """Returns a list of the shape of tensor, preferring static dimensions.

  Args:
    tensor: A tf.Tensor object to find the shape of.
    expected_rank: (optional) int. The expected rank of `tensor`. If this is
      specified and the `tensor` has a different rank, and exception will be
      thrown.
    name: Optional name of the tensor for the error message.

  Returns:
    A list of dimensions of the shape of tensor. All static dimensions will
    be returned as python integers, and dynamic dimensions will be returned
    as tf.Tensor scalars.
  """
  if name is None:
    name = tensor.name

  if expected_rank is not None:
    assert_rank(tensor, expected_rank, name)

  shape = tensor.shape.as_list()

  non_static_indexes = []
  for (index, dim) in enumerate(shape):
    if dim is None:
      non_static_indexes.append(index)

  if not non_static_indexes:
    return shape

  dyn_shape = tf.shape(tensor)
  for index in non_static_indexes:
    shape[index] = dyn_shape[index]
  return shape


def reshape_to_matrix(input_tensor):
  """Reshapes a >= rank 2 tensor to a rank 2 tensor (i.e., a matrix)."""
  ndims = input_tensor.shape.ndims
  if ndims < 2:
    raise ValueError("Input tensor must have at least rank 2. Shape = %s" %
                     (input_tensor.shape))
  if ndims == 2:
    return input_tensor

  width = input_tensor.shape[-1]
  output_tensor = tf.reshape(input_tensor, [-1, width])
  return output_tensor


def reshape_from_matrix(output_tensor, orig_shape_list):
  """Reshapes a rank 2 tensor back to its original rank >= 2 tensor."""
  if len(orig_shape_list) == 2:
    return output_tensor

  output_shape = get_shape_list(output_tensor)

  orig_dims = orig_shape_list[0:-1]
  width = output_shape[-1]

  return tf.reshape(output_tensor, orig_dims + [width])


def assert_rank(tensor, expected_rank, name=None):
  """Raises an exception if the tensor rank is not of the expected rank.

  Args:
    tensor: A tf.Tensor to check the rank of.
    expected_rank: Python integer or list of integers, expected rank.
    name: Optional name of the tensor for the error message.

  Raises:
    ValueError: If the expected shape doesn't match the actual shape.
  """
  if name is None:
    name = tensor.name

  expected_rank_dict = {}
  if isinstance(expected_rank, six.integer_types):
    expected_rank_dict[expected_rank] = True
  else:
    for x in expected_rank:
      expected_rank_dict[x] = True

  actual_rank = tensor.shape.ndims
  if actual_rank not in expected_rank_dict:
    scope_name = tf.get_variable_scope().name
    raise ValueError(
        "For the tensor `%s` in scope `%s`, the actual rank "
        "`%d` (shape = %s) is not equal to the expected rank `%s`" %
        (name, scope_name, actual_rank, str(tensor.shape), str(expected_rank)))



def scalar_attention(x, encode_dim, feature_dim, attention_dim, sequence_length=None,
                     mask_zero=False, maxlen=None, epsilon=1e-8, seed=0, scope_name="attention", reuse=False):
    """
    :param x: [batchsize, s, feature_dim]
    :param encode_dim: dim of encoder output
    :param feature_dim: dim of x (for self-attention, x is the encoder output;
                        for context-attention, x is the concat of encoder output and contextual info)
    :param sequence_length:
    :param mask_zero:
    :param maxlen:
    :param epsilon:
    :param seed:
    :param scope_name:
    :param reuse:
    :return: [batchsize, s, 1]
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        # W1: [feature_dim]
        W1 = tf.get_variable("W1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[feature_dim])
        # b1: [1]
        b1 = tf.get_variable("b1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[1])
    e = tf.einsum("bsf,f->bs", x, W1) + \
        tf.expand_dims(b1, axis=1)
    a = tf.exp(e)

    # apply mask after the exp. will be re-normalized next
    if mask_zero:
        # None * s
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.cast(mask, tf.float32)
        a = a * mask

    # in some cases especially in the early stages of training the sum may be almost zero
    s = tf.reduce_sum(a, axis=1, keep_dims=True)
    a /= tf.cast(s + epsilon, tf.float32)
    a = tf.expand_dims(a, axis=-1)

    return a

def vector_attention(x, encode_dim, feature_dim, attention_dim, sequence_length=None,
                     mask_zero=False, maxlen=None, epsilon=1e-8, seed=0,
                     scope_name="attention", reuse=False):
    """
    :param x: [batchsize, s, feature_dim]
    :param encode_dim: dim of encoder output
    :param feature_dim: dim of x (for self-attention, x is the encoder output;
                        for context-attention, x is the concat of encoder output and contextual info)
    :param attention_dim:attention hidden dim
    :param sequence_length:
    :param mask_zero:
    :param maxlen:
    :param epsilon:
    :param seed:
    :param scope_name:
    :param reuse:
    :return: [batchsize, s, encode_dim]
    """
    with tf.variable_scope(scope_name, reuse=reuse):
        # W1: [attention_dim, feature_dim]
        W1 = tf.get_variable("W1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[attention_dim, feature_dim])
        # b1: [attention_dim]
        b1 = tf.get_variable("b1_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[attention_dim])
        # W2: [encode_dim, attention_dim]
        W2 = tf.get_variable("W2_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[encode_dim, attention_dim])
        # b2: [encode_dim]
        b2 = tf.get_variable("b2_%s" % scope_name,
                             initializer=tf.truncated_normal_initializer(
                                 mean=0.0, stddev=0.2, dtype=tf.float32, seed=seed),
                             dtype=tf.float32,
                             shape=[encode_dim])
    # [batchsize, attention_dim, s]
    e = tf.nn.relu(
        tf.einsum("bsf,af->bas", x, W1) + \
        tf.expand_dims(tf.expand_dims(b1, axis=0), axis=-1))
    # [batchsize, s, encode_dim]
    e = tf.einsum("bas,ea->bse", e, W2) + \
        tf.expand_dims(tf.expand_dims(b2, axis=0), axis=0)
    a = tf.exp(e)

    # apply mask after the exp. will be re-normalized next
    if mask_zero:
        # [batchsize, s, 1]
        mask = tf.sequence_mask(sequence_length, maxlen)
        mask = tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
        a = a * mask

    # in some cases especially in the early stages of training the sum may be almost zero
    s = tf.reduce_sum(a, axis=1, keep_dims=True)
    a /= tf.cast(s + epsilon, tf.float32)

    return a

def _attend(x, sequence_length=None, method="ave", context=None, encode_dim=None,
            feature_dim=None, attention_dim=None, mask_zero=False, maxlen=None,
           bn=False, training=False, seed=0, scope_name="attention", reuse=False,
            num_heads=1):

    if method == "ave":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.reshape(mask, (-1, tf.shape(x)[1], 1))
            mask = tf.cast(mask, tf.float32)
            z = tf.reduce_sum(x * mask, axis=1)
            l = tf.reduce_sum(mask, axis=1)
            # in some cases especially in the early stages of training the sum may be almost zero
            epsilon = 1e-8
            z /= tf.cast(l + epsilon, tf.float32)
        else:
            z = tf.reduce_mean(x, axis=1)
    elif method == "sum":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.reshape(mask, (-1, tf.shape(x)[1], 1))
            mask = tf.cast(mask, tf.float32)
            z = tf.reduce_sum(x * mask, axis=1)
        else:
            z = tf.reduce_sum(x, axis=1)
    elif method == "max":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, (1, 1, tf.shape(x)[2]))
            masked_data = tf.where(tf.equal(mask, tf.zeros_like(mask)),
                                   tf.ones_like(x) * -np.inf, x)  # if masked assume value is -inf
            z = tf.reduce_max(masked_data, axis=1)
        else:
            z = tf.reduce_max(x, axis=1)
    elif method == "min":
        if mask_zero:
            # None * step_dim
            mask = tf.sequence_mask(sequence_length, maxlen)
            mask = tf.expand_dims(mask, axis=-1)
            mask = tf.tile(mask, (1, 1, tf.shape(x)[2]))
            masked_data = tf.where(tf.equal(mask, tf.zeros_like(mask)),
                                   tf.ones_like(x) * np.inf, x)  # if masked assume value is -inf
            z = tf.reduce_min(masked_data, axis=1)
        else:
            z = tf.reduce_min(x, axis=1)
    elif "attention" in method:
        if context is not None:
            y = tf.concat([x, context], axis=-1)
        else:
            y = x
        zs = []
        for i in range(num_heads):
            if "vector" in method:
                a = vector_attention(y, encode_dim, feature_dim, attention_dim, sequence_length, mask_zero, maxlen, seed=seed, scope_name=scope_name+str(i), reuse=reuse)
            else:
                a = scalar_attention(y, encode_dim, feature_dim, attention_dim, sequence_length, mask_zero, maxlen, seed=seed, scope_name=scope_name+str(i), reuse=reuse)
            zs.append(tf.reduce_sum(x * a, axis=1))
        z = tf.concat(zs, axis=-1)
    if bn:
        z = tf.layers.BatchNormalization()(z, training=training)

    return z


def attend(x, sequence_length=None, method="ave", context=None, encode_dim=None,
           feature_dim=None, attention_dim=None, mask_zero=False, maxlen=None,
           bn=False, training=False, seed=0, scope_name="attention", reuse=False,
           num_heads=1):
    if isinstance(method, list):
        outputs = [None]*len(method)
        for i,m in enumerate(method):
            outputs[i] = _attend(x, sequence_length, m, context, encode_dim, feature_dim, attention_dim, mask_zero, maxlen,
                                bn, training, seed, scope_name+m, reuse, num_heads)
        return tf.concat(outputs, axis=-1)
    else:
        return _attend(x, sequence_length, method, context, encode_dim, feature_dim, attention_dim, mask_zero, maxlen,
                                bn, training, seed, scope_name+method, reuse, num_heads)


def mlp_layer(input, fc_type, hidden_units, dropouts, scope_name, reuse=False, training=False, seed=0):
    if fc_type == "fc":
        output = dense_block(input, hidden_units=hidden_units, dropouts=dropouts,
                                         densenet=False, scope_name=scope_name,
                                         reuse=reuse,
                                         training=training, seed=seed)
    elif fc_type == "densenet":
        output = dense_block(input, hidden_units=hidden_units, dropouts=dropouts,
                                         densenet=True, scope_name=scope_name,
                                         reuse=reuse,
                                         training=training, seed=seed)
    elif fc_type == "resnet":
        output = resnet_block(input, hidden_units=hidden_units, dropouts=dropouts,
                                          cardinality=1, dense_shortcut=True, training=training,
                                          reuse=reuse,
                                          seed=seed,
                                          scope_name=scope_name)
    return output

def dense_block(x, hidden_units, dropouts, densenet=False, scope_name="dense_block", reuse=False, training=False, seed=0, bn=False):
    return _dense_block_mode1(x, hidden_units, dropouts, densenet, scope_name, reuse, training, seed, bn)

def _dense_block_mode1(x, hidden_units, dropouts, densenet=False, scope_name="dense_block", reuse=False, training=False, seed=0, bn=False):
    """
    :param x:
    :param hidden_units:
    :param dropouts:
    :param densenet: enable densenet
    :return:
    Ref: https://github.com/titu1994/DenseNet
    """
    for i, (h, d) in enumerate(zip(hidden_units, dropouts)):
        scope_name_i = "%s-dense_block_mode1-%s"%(str(scope_name), str(i))
        with tf.variable_scope(scope_name, reuse=reuse):
            z = tf.layers.dense(x, h, kernel_initializer=tf.glorot_uniform_initializer(seed=seed * i),
                                  reuse=reuse,
                                  name=scope_name_i)
            if bn:
                z = batch_normalization(z, training=training, name=scope_name_i+"-bn")
            z = tf.nn.relu(z)
            z = tf.layers.Dropout(d, seed=seed * i)(z, training=training) if d > 0 else z
            if densenet:
                x = tf.concat([x, z], axis=-1)
            else:
                x = z
    return x

def batch_normalization(x, training, name):
    # with tf.variable_scope(name, reuse=)
    bn_train = tf.layers.batch_normalization(x, training=True, reuse=None, name=name)
    bn_inference = tf.layers.batch_normalization(x, training=False, reuse=True, name=name)
    z = tf.cond(training, lambda: bn_train, lambda: bn_inference)
    return z

def resnet_block(input_tensor, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0,
                 scope_name="resnet_block", reuse=False):
    return _resnet_block_mode2(input_tensor, hidden_units, dropouts, cardinality, dense_shortcut, training, seed,
                               scope_name, reuse)

def _resnet_block_mode2(x, hidden_units, dropouts, cardinality=1, dense_shortcut=False, training=False, seed=0,
                        scope_name="_resnet_block_mode2", reuse=False):
    """A block that has a dense layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts

    xs = []
    # branch 0
    if dense_shortcut:
        with tf.variable_scope(scope_name, reuse=reuse):
            x0 = tf.layers.dense(x, h3, kernel_initializer=tf.glorot_uniform_initializer(seed * 1),
                                 bias_initializer=tf.zeros_initializer(),
                                 reuse=reuse,
                                 name=scope_name+"-dense-"+str("0"))
        xs.append(x0)
    else:
        xs.append(x)

    # branch 1 ~ cardinality
    for i in range(cardinality):
        xs.append(_resnet_branch_mode2(x, hidden_units, dropouts, training, seed, scope_name, reuse))

    x = tf.add_n(xs)
    return x

def _resnet_branch_mode2(x, hidden_units, dropouts, training=False, seed=0, scope_name="_resnet_branch_mode2", reuse=False):
    h1, h2, h3 = hidden_units
    dr1, dr2, dr3 = dropouts
    # name = "resnet"
    with tf.variable_scope(scope_name, reuse=reuse):
        # branch 2: bn-relu->weight
        x2 = tf.layers.BatchNormalization()(x)
        # x2 = batch_normalization(x, training=training, name=scope_name + "-bn-" + str(1))
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.Dropout(dr1)(x2, training=training) if dr1 > 0 else x2
        x2 = tf.layers.dense(x2, h1, kernel_initializer=tf.glorot_uniform_initializer(seed * 1),
                             bias_initializer=tf.zeros_initializer(),
                             name=scope_name+"-dense-"+str(1),
                             reuse=reuse)

        x2 = tf.layers.BatchNormalization()(x2)
        # x2 = batch_normalization(x2, training=training, name=scope_name + "-bn-" + str(2))
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.Dropout(dr2)(x2, training=training) if dr2 > 0 else x2
        x2 = tf.layers.dense(x2, h2, kernel_initializer=tf.glorot_uniform_initializer(seed * 2),
                             bias_initializer=tf.zeros_initializer(),
                             name=scope_name + "-dense-" + str(2),
                             reuse=reuse)

        x2 = tf.layers.BatchNormalization()(x2)
        # x2 = batch_normalization(x2, training=training, name=scope_name + "-bn-" + str(3))
        x2 = tf.nn.relu(x2)
        x2 = tf.layers.Dropout(dr3)(x2, training=training) if dr3 > 0 else x2
        x2 = tf.layers.dense(x2, h3, kernel_initializer=tf.glorot_uniform_initializer(seed * 3),
                             bias_initializer=tf.zeros_initializer(),
                             name=scope_name + "-dense-" + str(3),
                             reuse=reuse)

    return x2

def _textcnn(x, conv_op, num_filters=8, filter_sizes=[2, 3], bn=False, training=False,
            timedistributed=False, scope_name="textcnn", reuse=False, activation=tf.nn.relu):
    # x: None * step_dim * embed_dim
    conv_blocks = []
    for i, filter_size in enumerate(filter_sizes):
        scope_name_i = "%s_textcnn_%s"%(str(scope_name), str(filter_size))
        with tf.variable_scope(scope_name_i, reuse=reuse):
            if timedistributed:
                input_shape = tf.shape(x)
                step_dim = input_shape[1]
                embed_dim = input_shape[2]
                x = tf.transpose(x, [0, 2, 1])
                # None * embed_dim * step_dim
                x = tf.reshape(x, [input_shape[0] * embed_dim, step_dim, 1])
                conv = conv_op(
                    inputs=x,
                    filters=1,
                    kernel_size=filter_size,
                    padding="same",
                    activation=activation,
                    strides=1,
                    reuse=reuse,
                    name=scope_name_i)
                conv = tf.reshape(conv, [input_shape[0], embed_dim, step_dim])
                conv = tf.transpose(conv, [0, 2, 1])
            else:
                conv = conv_op(
                    inputs=x,
                    filters=num_filters,
                    kernel_size=filter_size,
                    padding="same",
                    activation=activation,
                    strides=1,
                    reuse=reuse,
                    name=scope_name_i)
            if bn:
                conv = tf.layers.BatchNormalization()(conv, training)
            # conv = activation(conv)
            conv_blocks.append(conv)
    if len(conv_blocks) > 1:
        z = tf.concat(conv_blocks, axis=-1)
    else:
        z = conv_blocks[0]
    return z


def textcnn(x, num_layers=2, num_filters=8, filter_sizes=[2, 3], bn=False, training=False,
            timedistributed=False, scope_name="textcnn", reuse=False, activation=tf.nn.relu,
            gated_conv=False, residual=False):
    if gated_conv:
        if residual:
            conv_op = residual_gated_conv1d_op
        else:
            conv_op = gated_conv1d_op
    else:
        conv_op = tf.layers.conv1d
    conv_blocks = []
    for i in range(num_layers):
        scope_name_i = "%s_textcnn_layer_%s" % (str(scope_name), str(i))
        x = _textcnn(x, conv_op, num_filters, filter_sizes, bn, training, timedistributed, scope_name_i, reuse, activation)
        conv_blocks.append(x)
    if len(conv_blocks) > 1:
        z = tf.concat(conv_blocks, axis=-1)
    else:
        z = conv_blocks[0]
    return z

def gated_conv1d_op(inputs, filters=8, kernel_size=3, padding="same", activation=None, strides=1, reuse=False, name=""):
    conv_linear = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        strides=strides,
        reuse=reuse,
        name=name+"_linear")
    conv_gated = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.sigmoid,
        strides=strides,
        reuse=reuse,
        name=name+"_gated")
    conv = conv_linear * conv_gated
    return conv


def residual_gated_conv1d_op(inputs, filters=8, kernel_size=3, padding="same", activation=None, strides=1, reuse=False, name=""):
    conv_linear = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=None,
        strides=strides,
        reuse=reuse,
        name=name+"_linear")
    conv_gated = tf.layers.conv1d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        padding="same",
        activation=tf.nn.sigmoid,
        strides=strides,
        reuse=reuse,
        name=name+"_gated")
    conv = inputs * (1. - conv_gated) + conv_linear * conv_gated
    return conv