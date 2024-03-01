import muax
from muax import nn
import gym_trading_env
import gymnasium as gym

import pandas as pd
# Available in the github repo : examples/data/BTC_USD-Hourly.csv
url = "https://raw.githubusercontent.com/ClementPerroud/Gym-Trading-Env/main/examples/data/BTC_USD-Hourly.csv"
df = pd.read_csv(url, parse_dates=["date"], index_col= "date")
df.sort_index(inplace= True)
df.dropna(inplace= True)
df.drop_duplicates(inplace=True)

df["feature_close"] = df["close"].pct_change()

# Create the feature : open[t] / close[t]
df["feature_open"] = df["open"]/df["close"]

# Create the feature : high[t] / close[t]
df["feature_high"] = df["high"]/df["close"]

# Create the feature : low[t] / close[t]
df["feature_low"] = df["low"]/df["close"]

 # Create the feature : volume[t] / max(*volume[t-7*24:t+1])
df["feature_volume"] = df["Volume USD"] / df["Volume USD"].rolling(7*24).max()

df.dropna(inplace= True) # Clean again !

support_size = 10 
embedding_size = 8
discount = 0.99
num_actions = 3
full_support_size = int(support_size * 2 + 1)

repr_fn = nn._init_representation_func(nn.Representation, embedding_size)
pred_fn = nn._init_prediction_func(nn.Prediction, num_actions, full_support_size)
dy_fn = nn._init_dynamic_func(nn.Dynamic, embedding_size, num_actions, full_support_size)

tracer = muax.PNStep(10, discount, 0.5)
buffer = muax.TrajectoryReplayBuffer(500)

gradient_transform = muax.model.optimizer(init_value=0.02, peak_value=0.02, end_value=0.002, warmup_steps=5000, transition_steps=5000)

model = muax.MuZero(repr_fn, pred_fn, dy_fn, policy='muzero', discount=discount,
                    optimizer=gradient_transform, support_size=support_size)
env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )
test_env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate= 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
    )
model_path = muax.fit(model, env=env,
                    test_env=test_env, 
                    max_episodes=1000,
                    max_training_steps=10000,
                    tracer=tracer,
                    buffer=buffer,
                    k_steps=10,
                    sample_per_trajectory=1,
                    num_trajectory=32,
                    tensorboard_dir='./tensorboard/cartpole',
                    model_save_path='./models/cartpole',
                    save_name='cartpole_model_params',
                    random_seed=0,
                    log_all_metrics=True)