#!/usr/bin/env python
# coding=utf-8
"""
@Author : yonas
@File   : util.py
"""
import platform

IS_LINUX = 'Linux' in platform.system()
if IS_LINUX:
    import readline

import tensorflow as tf
hparams = tf.contrib.training.HParams(
    batch_size=8,
    worddict='data/word.stat',
    train_data_dir='data/chit_chat-train*',

    model_dir='model',
    save_steps=5000,
    max_steps=500000,
    max_to_keep=2,

    vacab_size=4255,
    hidden_size=128,
    filter_size=512,
    num_encoder_layers=3,
    num_decoder_layers=2,
    num_heads=8,

    keep_space=False,
    stop_nums=None,

    sampling_method='argmax',
    sampling_temp=1.0,

    layer_postprocess_sequence='da',
    layer_preprocess_sequence='n',
    layer_prepostprocess_dropout=0,
    symbol_dropout=0,
    attention_dropout=0,
    relu_dropout=0,

    initializer='uniform_unit_scaling',
    initializer_gain=1.0,
    weight_decay=0,
    weight_noise=0.0,
    multiply_embedding_mode='sqrt_depth',
    shared_embedding=True,
    shared_embedding_and_softmax_weights=True,

    clip_grad_norm=0.0,
    grad_noise_scale=0.0,

    max_input_seq_length=50,
    max_target_seq_length=50,

    learning_rate=0.2,
    learning_rate_constant=2.0,
    learning_rate_cosine_cycle_steps=250000,
    learning_rate_decay_rate=1.0,
    learning_rate_decay_scheme='noam',
    learning_rate_decay_staircase=False,
    learning_rate_decay_steps=5000,
    learning_rate_minmum=None,
    learning_rate_schedule='constant*linear_warmup*rsqrt_decay*rsqrt_hidden_size',
    learning_rate_warmup_steps=16000,

    optimizer='Adam',  # or 'Momentum'
    optimizer_adam_beta1=0.9,
    optimizer_adam_beta2=0.997,
    optimizer_adam_epsilon=1e-09,
    optimizer_momentum_momentum=0.9,
    optimizer_momentum_nesterov=False,
    optimizer_multistep_accumulate_steps=None,

    summarize_grads=False,
    summarize_vars=False,




)