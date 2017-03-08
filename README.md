# runexp

A simple tool to run experiments in a complex workflow

## Steps to run experiments

1. Copy `runexp.py` into your directory.
2. Write a script to specify a workflow.  The script has to import `runexp.Workflow`, specify tasks using `Workflow`, and execute `Workflow.run()`.
3. Run the script.

## Example

The following script runs `ls -l`, sorts its output, and takes the first line of the sorted result.

```python
import runexp
exp = runexp.Workflow()
exp(target='input.txt', rule='ls -l > input.txt')
exp(source='input.txt', target='sorted.txt', rule='sort input.txt > sorted.txt')
exp(source='sorted.txt', target='head.txt', rule='head -n 1 sorted.txt > head.txt')
exp.run()
```

