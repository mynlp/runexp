# coding: utf-8
"""A simple example of runexp

`python example.py` will run a simple example (example1).
It runs `ls -l`, sorts its output, and outputs the first line of the sorted result to `head.txt`.

`python example.py example2` and `python example.py example3` will run other examples.

`python example.py clean` will clean up the files and the directory created by the above examples.
"""

import runexp
exp = runexp.Workflow(goal_targets=['example1'])

exp(target='input.txt', rule='ls -l > input.txt')
exp(source='input.txt', target='sorted.txt', rule='sort input.txt > sorted.txt')
exp(source='sorted.txt', target='head.txt', rule='head -n 1 sorted.txt > head.txt')
exp(source='head.txt', target='example1', phony=True)

exp(target='date1.txt', rule='sleep 3; date > date1.txt')
exp(target='date2.txt', rule='sleep 3; date > date2.txt')
exp(source=['date1.txt', 'date2.txt'], target='merged.txt', rule='cat date1.txt date2.txt > merged.txt')
exp(source='merged.txt', target='example2', phony=True)

exp(target='tmp_dir', rule='mkdir -p tmp_dir')
exp(source='runexp.py', target='tmp_dir/copy.txt', require='tmp_dir', rule='cat runexp.py > tmp_dir/copy.txt')
exp(source='tmp_dir/copy.txt', target='example3', phony=True)

exp(target='clean', rule=['rm -f *.txt', 'rm -rf tmp_dir'], no_exec=True, phony=True)

exp.run()

