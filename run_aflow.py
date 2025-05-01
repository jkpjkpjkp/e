#!/usr/bin/env python3
import os
import sys

# Create a symlink from vizwiz.py to viswiz.py if it doesn't exist
if not os.path.exists('vizwiz.py') and os.path.exists('viswiz.py'):
    print("Creating symlink from vizwiz.py to viswiz.py")
    os.symlink('viswiz.py', 'vizwiz.py')

# Run aflow.py with the vizwiz dataset
os.system('python aflow.py --dataset vizwiz --seed_file seed1.py')
