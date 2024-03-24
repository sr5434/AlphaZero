import mctx
import haiku as hk
import jax.numpy as jnp
import jax
import pgx
from pydantic import BaseModel
import pickle
from pgx.experimental import auto_reset
from typing import NamedTuple

class Config(BaseModel):
    env_id: pgx.EnvId = ""
    seed: int = 0
    max_num_iters: int = 1
    # network params
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    #Testing
    selfplay_batch_size: int = 1028
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"

with open(input("Path to checkpoint: "), "rb") as f:
    d = pickle.load(f)
    print(d.keys())
    model = d["model"]
    config = d["config"]

env = pgx.make(config.env_id)

class BlockV2(hk.Module):
    def __init__(self, num_channels, name="BlockV2"):
        super(BlockV2, self).__init__(name=name)
        self.num_channels = num_channels

    def __call__(self, x, is_training, test_local_stats):
        i = x
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)
        x = hk.Conv2D(self.num_channels, kernel_shape=3)(x)
        return x + i


class AZNet(hk.Module):
    """AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions,
        num_channels: int = 64,
        num_blocks: int = 5,
        name="az_net",
    ):
        super().__init__(name=name)
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_cls = BlockV2

    def __call__(self, x, is_training, test_local_stats):
        if config.env_id == "kuhn_poker" or config.env_id == "leduc_holdem":
            x = x.reshape((x.shape[0], x.shape[1], 1))
        x = x.astype(jnp.float32)
        x = hk.Conv2D(self.num_channels, kernel_shape=2)(x)

        for i in range(self.num_blocks):
            x = self.resnet_cls(self.num_channels, name=f"block_{i}")(
                x, is_training, test_local_stats
            )
        x = hk.BatchNorm(True, True, 0.9)(x, is_training, test_local_stats)
        x = jax.nn.relu(x)

        # policy head
        logits = hk.Conv2D(output_channels=2, kernel_shape=1)(x)
        logits = hk.BatchNorm(True, True, 0.9)(logits, is_training, test_local_stats)
        logits = jax.nn.relu(logits)
        logits = hk.Flatten()(logits)
        logits = hk.Linear(self.num_actions)(logits)

        # value head
        v = hk.Conv2D(output_channels=1, kernel_shape=1)(x)
        v = hk.BatchNorm(True, True, 0.9)(v, is_training, test_local_stats)
        v = jax.nn.relu(v)
        v = hk.Flatten()(v)
        v = hk.Linear(self.num_channels)(v)
        v = jax.nn.relu(v)
        v = hk.Linear(1)(v)
        v = jnp.tanh(v)
        v = v.reshape((-1,))

        return logits, v

def forward_fn(x, is_eval=False):
    net = AZNet(
        num_actions=env.num_actions,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
    )
    policy_out, value_out = net(x, is_training=not is_eval, test_local_stats=False)
    return policy_out, value_out

forward = hk.without_apply_rng(hk.transform_with_state(forward_fn))

model_params, model_state = model
init_fn = jax.jit(jax.vmap(env.init))
step_fn = jax.jit(jax.vmap(env.step))

states = []
batch_size = 1
keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
state = init_fn(keys)
states.append(state)

while not (state.terminated | state.truncated).all():
    (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
    action = logits.argmax(axis=-1)
    state = step_fn(state, action)
    states.append(state)
pgx.save_svg_animation(states, "game.svg", frame_duration_seconds=0.5)