# Reinforcement Learning CTF Agents

## To Run

To run a game against the baseline team, run:

```
python capture.py -r baselineTeam -b baselineTeam -l RANDOM13
```

To train, run:

```
python train.py 100
```

This will train myTeam vs itself on random maps 100 times, saving it's updated weights to
file after every decision.  
