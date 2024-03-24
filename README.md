# AlphaZero
Implementing the AlphaZero algorithm(but with the MuZero Gumbel policy) for multiple games with PGX and MCTX. Thanks to Google's TensorFlow Research Cloud for providing compute resources. The training script is modified from the sample AlphaZero script provided in the PGX GitHub Repo.
Scripts:
 - ```main.py```: AlphaZero for playing games(Atari, Go, Poker, etc.)
 - ```test.py```: Test the model and output an SVG of it playing itself.


Checkpoints(more coming soon):
 - [Kuhn Poker](https://huggingface.co/sr5434/AlphaZero-Kuhn-Poker)
 - [Othello/Reversi(Work in progress)](http://huggingface.co/sr5434/alphazero-othello)

Othello model plays itself(step 15145):


![AI plays itself in game of Othello](./game.svg)

