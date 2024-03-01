"""Test MuZero"""
import muax
from muax import nn 
import gymnasium as gym
import jax


support_size = 10 
embedding_size = 8
discount = 0.99
num_actions = 2
full_support_size = int(support_size * 2 + 1)

repr_fn = nn._init_representation_func(nn.Representation, embedding_size)
pred_fn = nn._init_prediction_func(nn.Prediction, num_actions, full_support_size)
dy_fn = nn._init_dynamic_func(nn.Dynamic, embedding_size, num_actions, full_support_size)

from muax.test import test
gradient_transform = muax.model.optimizer(init_value=0.02, peak_value=0.02, end_value=0.002, warmup_steps=5000, transition_steps=5000)
model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                    optimizer=gradient_transform, support_size=support_size)

model.load("./cartpole_model_params.npy")

env_id = 'CartPole-v1'
test_env = gym.make(env_id, render_mode='rgb_array')
test_key = jax.random.PRNGKey(0)
print(test(model, test_env, test_key, num_simulations=50, num_test_episodes=100, random_seed=None))