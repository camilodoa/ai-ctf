from os import system
import sys
import signal


def run(epochs):
    # Run game of myTeam vs itself on a random map
        for i in xrange(epochs):
            trial = system("python capture.py -r myTeam -b myTeam -l RANDOM")
            if trial == signal.SIGINT:
                return
                break

if __name__ == "__main__":
    train_epochs = sys.argv[0] if len(sys.argv) == 1 else 1
    run(train_epochs)
