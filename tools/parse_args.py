import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--rI', dest='intracortical_scale', type=float, required=True)
parser.add_argument('--rC', dest='LGNinput_scale', type=float,required=True)
args = parser.parse_args()
print("args",args)
