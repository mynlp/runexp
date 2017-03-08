# coding: utf-8
"""A simple example of runexp
This script runs `ls -l`, sorts its output, and takes the first line of the sorted result.
"""

import runexp
exp = runexp.Workflow()
exp(target='input.txt', rule='ls -l > input.txt')
exp(source='input.txt', target='sorted.txt', rule='sort input.txt > sorted.txt')
exp(source='sorted.txt', target='head.txt', rule='head -n 1 sorted.txt > head.txt')
exp.run()

