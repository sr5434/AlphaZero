# AlphaZero
Implementing the AlphaZero algorithm(albeit with the MuZero Gumbel policy) for multiple games with PGX and MCTX, as well as MuZero for stock trading with Muax. Thanks to Google's TensorFlow Research Cloud for providing compute resources. The training script is modified from the sample AlphaZero script provided in the PGX GitHub Repo.
Scripts:
 - ```main.py```: AlphaZero for games(Atari, Go, Poker, etc.)
 - ```muzero_trading.py```: MuZero for BTC trading(with Muax)
 - ```test.py```: Test the trained Bitcoin Trader Bot
 - ```test_model.py```: Test the games models(i.e. Kuhn Poker)


Checkpoints(more coming soon):
 - [Kuhn Poker](https://huggingface.co/sr5434/AlphaZero-Kuhn-Poker)
