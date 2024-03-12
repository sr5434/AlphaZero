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
    env_id: pgx.EnvId = input("EnvID:")
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
    
    class Config:
        extra = "forbid"

config: Config = Config()
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
with open(input("Path to checkpoint: "), "rb") as f:
        d = pickle.load(f)
        model = d["model"]


def recurrent_fn(model, rng_key: jnp.ndarray, action: jnp.ndarray, state: pgx.State):
    # model: params
    # state: embedding
    if config.env_id not in (
        "minatar-asterix",
        "minatar-breakout",
        "minatar-freeway",
        "minatar-seaquest",
        "minatar-space_invaders",
        "2048"
    ):
        del rng_key
    if config.env_id in (
        "minatar-asterix",
        "minatar-breakout",
        "minatar-freeway",
        "minatar-seaquest",
        "minatar-space_invaders",
        "2048"
    ):
        step_fn = jax.vmap(env.step)
        keys = jax.random.split(rng_key, state.observation.shape[0])
    model_params, model_state = model

    current_player = state.current_player
    if config.env_id in (
        "minatar-asterix",
        "minatar-breakout",
        "minatar-freeway",
        "minatar-seaquest",
        "minatar-space_invaders",
        "2048"
    ):
        state = step_fn(state, action, keys)
    else:
        state = jax.vmap(env.step)(state, action)
    (logits, value), _ = forward.apply(model_params, model_state, state.observation, is_eval=True)
    # mask invalid actions
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = -1.0 * jnp.ones_like(value)
    discount = jnp.where(state.terminated, 0.0, discount)

    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray


@jax.pmap
def selfplay(model, rng_key: jnp.ndarray) -> SelfplayOutput:
    model_params, model_state = model
    batch_size = config.selfplay_batch_size
    def step_fn(state, key) -> SelfplayOutput:
        key1, key2 = jax.random.split(key)
        observation = state.observation

        (logits, value), _ = forward.apply(
            model_params, model_state, state.observation, is_eval=True
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)

        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        state = jax.vmap(auto_reset(env.step, env.init))(state, policy_output.action, keys)
        discount = -1.0 * jnp.ones_like(value)
        discount = jnp.where(state.terminated, 0.0, discount)
        return state, SelfplayOutput(
            obs=observation,
            action_weights=policy_output.action_weights,
            reward=state.rewards[jnp.arange(state.rewards.shape[0]), actor],
            terminated=state.terminated,
            discount=discount,
        )

    # Run selfplay for max_num_steps by batch
    rng_key, sub_key = jax.random.split(rng_key)
    keys = jax.random.split(sub_key, batch_size)
    state = jax.vmap(env.init)(keys)
    key_seq = jax.random.split(rng_key, 1)
    _, data = jax.lax.scan(step_fn, state, key_seq)

    return data

model = jax.device_put_replicated((model), jax.local_devices())
rng_key, subkey = jax.random.split(jax.random.PRNGKey(0))
keys = jax.random.split(subkey, 1)
print(selfplay(model, keys).reward)