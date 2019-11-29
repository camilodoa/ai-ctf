from os import system
import sys
import signal
import subprocess
import random


def train(epochs):
    # Run game of myTeam vs itself on a random map
        for i in xrange(epochs):
            try:
                subprocess.call(['python2', 'capture.py', '-r', 'myTeam', '-b', 'myTeam', '-l', 'RANDOM', '-q'])
            except KeyboardInterrupt:
                break
                return

def validate():
    # Test current model against baselineTeam agents
    try:
        teams = ["myTeam", "baselineTeam"]
        red = random.choice(teams)
        blue = random.choice(teams)
        while(red == blue):
            blue = random.choice(teams)
        subprocess.call(['python2', 'capture.py', '-r', red, '-b', blue, '-l', 'RANDOM'])
    except KeyboardInterrupt:
        return



if __name__ == "__main__":
    train_epochs = int(sys.argv[1]) if len(sys.argv) == 2 else 1
    print("training for ", train_epochs)
    # train(train_epochs)
    print("validating")
    validate()
