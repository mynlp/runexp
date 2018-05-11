# coding: utf-8
"""A simple framework to run experiments with a complex workflow

- Features
  - Users define *tasks*, which specify source/target files and a
    command to generate target files
  - The system automatically computes dependencies among tasks and
    execute them in order
  - Tasks are executed only when:
    - some targets do not exist
    - some targets are older than source files (in a similar way as
      `make`), or
    - some sources are rebuilt in preceding tasks
  - The system runs tasks in parallel

- How to use
  1. Create an instance of `Workflow`
  2. Call this instance with argument source, target, rule, etc.
  3. Call the method `run()`
  - See the example below.  A more elaborated example is found in the
    end of "runexp.py".  You can see what happens by running:
    `python runexp.py`

- Example
  - The following example executes `ls -l`, `sort`, and then `head`.
  - The order of `exp()` does not matter.  The system automatically
    computes dependencies among tasks, and executes them in the
    appropriate order.

```
import runexp
exp = runexp.Workflow()
exp(target='input.txt', rule='ls -l > input.txt')
exp(source='input.txt', target='sorted.txt', rule='sort input.txt > sorted.txt')
exp(source='sorted.txt', target='head.txt', rule='head -n 1 sorted.txt > head.txt')
exp.run()
```

- More features
  - Output dependencies of tasks into a PNG file to visualize
  - Users can specify fine-grained conditions to execute a task
    - `always`: always execute a task
    - `no_timestamp`: do not check timestamp (i.e. do not execute a
      task even when the targets are old)
    - these specifications can also be overridden by the command-line
      argument
  - Users can specify any dependent files in addition to sources
    (e.g. a script to generate targets)
    - Tasks are executed when any dependent files are newer than
      targets (e.g. the script is modified)
    - Dependent files are used only for checking timestamps; not used
      for computing task dependencies
  - Users can define environment variables for each worker to execute
    tasks
    - can be used to change the behavior of each worker in parallel
      processing
    - e.g. allocate a dedicated GPU to each worker
  - Users can specify resource conditions on workers and tasks
    - Tasks are assigned to workers that have sufficient resources to
      run the task
    - e.g. assign GPUs to each worker, and run a task that requires
      GPUs on the worker with a sufficient number of GPUs

- Defining tasks
  - You can add a task to the workflow by calling the workflow
    instance with the following arguments.
    - source: input files (space-separated string or list of strings)
    - target: target files (space-separated string or list of strings)
    - rule: command to execute (string)
    - depend (optional): other dependent files (space-separated string
      or list of strings)
    - name (optional): a short name shown in the log message
  - The following options maybe specified to control the behavior:
    - always: always execute this task (bool; default=False)
    - no_timestamp: do not check timestamp (bool; default=False)
    - ignore_same_task: ignore doubly added equivalent tasks (bool;
      default=False); when equivalent tasks are found but this option
      is False, the system shows an error.

- Defining resource conditions
  - Specify available resources for workers, and required resources for tasks
  - Available resources for workers
    - Use "-r" option to give a list of available resources for
      workers.  Each element is a dict, which denotes available
      resources for a worker.
    - You can also use "resources" argument of the constructor or
      'set_options()' of Workflow
    - e.g. the following example denotes that the first worker has one
      GPU and 4GB memory, while the second worker has no GPU and 16GB
      memory.
      - `exp = Workflow(resources=[{'GPU': 1, 'Mem': 4}, {'Mem': 16}])
  - Required resources for tasks
    - Specify required resources for a task by "resource" argument
    - e.g. the following example specifies that the task requires 8GB
      memory.
      - `exp(resource={'Mem': 8}, ...)`
"""

from __future__ import print_function, unicode_literals
import sys
import os
import subprocess
import argparse
import logging
import multiprocessing
from datetime import datetime
from collections import deque
from fnmatch import fnmatch
import signal
import json
import psutil
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO

if sys.version_info.major == 2:
    def isstr(s):
        return isinstance(s, basestring)
    def iterdict(d):
        return d.iteritems()
else:
    def isstr(s):
        return isinstance(s, str)
    def iterdict(d):
        return d.items()

logger = logging.getLogger(__name__)
#logging.basicConfig(level=logging.DEBUG, format='%(asctime)s:%(name)s:%(funcName)s:%(levelname)s: %(message)s')
logging.basicConfig(level=logging.INFO, format='%(asctime)s:%(name)s:%(levelname)s: %(message)s')

######################################################################

def coloring(color, text):
    """Print a text in a specified color"""
    color_sequences = {
        'default': '\033[0m',
        'black': '\033[30m',
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'purple': '\033[35m',
        'lightblue': '\033[36m',
        'white': '\033[37m',
        }
    return color_sequences[color] + text + color_sequences['default']
    
######################################################################

class Task:
    """A single task to receive source files and produce target files"""
    def __init__(self, name=None, desc=None, source=[], target=[], rule=[], depend=[], resource={}, always=False, no_timestamp=False, ignore_same_task=False):
        if isstr(source):
            source = source.split()
        if isstr(target):
            target = target.split()
        if isstr(rule):
            rule = rule.split('\n')
        if isstr(depend):
            depend = depend.split()
        if name is None:
            name = '; '.join(rule)
        if desc is None:
            desc = name
        if not isinstance(source, list):
            raise ValueError("source must be a string or a list of strings: {}".format(source))
        if not isinstance(target, list):
            raise ValueError("target must be a string or a list of strings: {}".format(target))
        if not isinstance(rule, list):
            raise ValueError("rule must be a string or a list of strings: {}".format(rule))
        if not isinstance(depend, list):
            raise ValueError("depend must be a string or a list of strings: {}".format(depend))
        if not isinstance(resource, dict):
            raise ValueError("resource must be dict")
        self.name = name
        self.desc = desc
        self.source = source
        self.target = target
        self.rule = rule
        self.depend = depend
        self.resource = resource
        self.always = always  # force all commands to be executed
        self.no_timestamp = no_timestamp  # run commands only when targets do not exist (do not check timestamp)
        self.ignore_same_task = ignore_same_task
        pass

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return u'Task {}: rule="{}" source=[{}] target=[{}] depend=[{}]'.format(self.name, '; '.join(self.rule), ','.join(self.source), ','.join(self.target), ','.join(self.depend))

    def resource_satisfied(self, available_resource):
        """Check whether all required resources are satisfied"""
        assert(isinstance(available_resource, dict))
        for name, value in iterdict(self.resource):
            if name not in available_resource:
                return False
            if isinstance(value, int) or isinstance(value, float):
                if not value <= available_resource[name]:
                    return False
            elif isstr(value):
                if value != available_resource[name]:
                    return False
            else:
                raise ValueError("Resource value must be int, float, or string: %s", value)
        return True
    
    def show_rule(self):
        return '; '.join(self.rule)

    def show_task(self):
        depend = "  depend: {}\n".format(' '.join(self.depend)) if len(self.depend) != 0 else ""
        options = [x[1] for x in zip([self.always, self.no_timestamp, self.ignore_same_task], ["always", "no_timestamp", "ignore_same_task"]) if x[0]]
        options_str = "  options: {}\n".format(', '.join(options)) if len(options) > 0 else ""
        return """Task: {}
  source: {}
  target: {}
  rule: {}
{}{}""".format(self.name, ' '.join(self.source), ' '.join(self.target), '; '.join(self.rule), depend, options_str)
    
######################################################################

class TaskGraph:
    """Construct a graph of dependencies from task definitions"""

    def __init__(self, task_list, targets=None, always=False, no_timestamp=False):
        if not isinstance(task_list, list) or not all([isinstance(t, Task) for t in task_list]):
            raise ValueError("task_list must be list of Task")
        logger.debug('TaskGraph.task_list: %s', task_list)
        self.task_list = task_list
        # task dependency graph
        self.goal_targets = targets
        self.always = always  # run all tasks including up-to-date tasks
        self.no_timestamp = no_timestamp  # run tasks only when targets do not exist (do not check timestamp)
        self.prev_tasks = None
        self.next_tasks = None
        self.initial_tasks = None   # tasks to begin with
        self.executed_tasks = None   # tasks to be executed, including up-to-date tasks
        self.outdated_tasks = None  # outdated tasks -> needs to be executed
        self.__make_dependencies()
        pass

    def __make_prev_next_tasks(self):
        """Set prev_tasks and next_tasks by following source/target"""
        logger.debug('TaskGraph.__make_prev_next_tasks()')
        prev_tasks = {}
        next_tasks = {}
        # compute prev/next tasks for each file
        for task_id, task in enumerate(self.task_list):
            for source in task.source:
                source_path = os.path.realpath(source)  # use realpath to process same files with different paths
                next_tasks[source_path] = next_tasks.get(source_path, []) + [task_id]
            for target in task.target:
                target_path = os.path.realpath(target)  # use realpath to process same files with different paths
                prev_tasks[target_path] = prev_tasks.get(target_path, []) + [task_id]
        # compute prev/next tasks for each task
        self.prev_tasks = [[] for _ in range(self.num_tasks())]
        self.next_tasks = [[] for _ in range(self.num_tasks())]
        for task_id, task in enumerate(self.task_list):
            for source in task.source:
                source_path = os.path.realpath(source)
                self.prev_tasks[task_id].extend(prev_tasks.get(source_path, []))
            for target in task.target:
                target_path = os.path.realpath(target)
                self.next_tasks[task_id].extend(next_tasks.get(target_path, []))
        self.prev_tasks = [list(set(l)) for l in self.prev_tasks]
        self.next_tasks = [list(set(l)) for l in self.next_tasks]
        logger.debug('prev_tasks: %s', self.prev_tasks)
        logger.debug('next_tasks: %s', self.next_tasks)
        pass
    
    def __traverse_backwards(self):
        """Traverse tasks from goal targets and obtain previous tasks and tasks to be executed"""
        logger.debug('TaskGraph.__traverse_backwards()')
        if self.goal_targets is None or len(self.goal_targets) == 0:
            # all tasks will be run
            logger.debug('TaskGraph.__traverse_backwards(): targets are not specified.  all targets will be run')
            self.executed_tasks = set(range(self.num_tasks()))
            return
        # traverse dependencies from goal targets
        logger.debug('TaskGraph.__traverse_backwards(): traverse dependencies from goal targets: %s', self.goal_targets)
        target_tasks = set()
        for target in self.goal_targets:
            tasks = [task_id for task_id, task in enumerate(self.task_list) if any([fnmatch(t, target) for t in task.target]) ]
            if len(tasks) == 0:
                raise ValueError('Target not found in task definitions: ' + target)
            target_tasks |= set(tasks)
        prev_tasks = target_tasks
        while True:
            new_tasks = set(sum([self.prev_tasks[task_id] for task_id in prev_tasks], []))
            if len(new_tasks) == 0: break
            target_tasks |= new_tasks
            prev_tasks = new_tasks
        self.executed_tasks = target_tasks
        logger.debug('TaskGraph.__traverse_backwards(): executed_tasks: %s', list(self.executed_tasks))
        pass

    def __set_initial_tasks(self):
        """Obtain initial tasks to begin with"""
        logger.debug('TaskGraph.__set_initial_tasks()')
        # initial task = tasks with empty prev_task
        self.initial_tasks = { task_id for task_id in self.executed_tasks if len(self.prev_tasks[task_id]) == 0 }
        logger.debug('TaskGraph.__set_initial_tasks(): initial_tasks: %s', self.initial_tasks)
        pass
    
    def __up_to_date(self, task):
        # always run the task
        if self.always or task.always: return False
        # no targets -> always up-to-date
        if len(task.target) == 0: return True
        # no source -> up-to-date if all targets exists
        if len(task.source) == 0 and len(task.depend) == 0:
            return all([os.path.exists(target) for target in task.target])
        # check timestamp
        source_timestamps = [os.stat(f).st_mtime for f in task.source + task.depend if os.path.exists(f)]
        target_timestamps = [os.stat(f).st_mtime for f in task.target if os.path.exists(f)]
        if len(source_timestamps) != len(task.source) + len(task.depend) or len(target_timestamps) != len(task.target):
            # some source/target files do not exist -> not up-to-date
            return False
        if self.no_timestamp or task.no_timestamp:
            # do not check timestamp
            return True
        # check timestamp to judge
        return max(source_timestamps) <= min(target_timestamps)
    
    def __check_outdated_tasks(self):
        """Check outdated tasks"""
        logger.debug('TaskGraph.__check_outdated_tasks()')
        outdated_tasks = set()
        for task_id, task in enumerate(self.task_list):
            # task is added to the outdated_tasks if targets are not up-to-date
            if not self.__up_to_date(task):
                outdated_tasks.add(task_id)
        prev_outdated_tasks = outdated_tasks
        while True:
            new_outdated_tasks = set(sum([self.next_tasks[task_id] for task_id in prev_outdated_tasks], []))
            if len(new_outdated_tasks) == 0: break
            outdated_tasks |= new_outdated_tasks
            prev_outdated_tasks = new_outdated_tasks
        self.outdated_tasks = outdated_tasks
        logger.debug('TaskGraph.__check_outdated_tasks(): outdated_tasks: %s', self.outdated_tasks)
        pass

    def __make_dependencies(self):
        """Make task dependency graph from task list and godl targets"""
        logger.debug('TaskGraph.make_dependencies()')
        # compute previous/next tasks
        self.__make_prev_next_tasks()
        # obtain tasks that must be executed to build targets
        self.__traverse_backwards()
        # set initial tasks
        self.__set_initial_tasks()
        # check up-to-date tasks
        self.__check_outdated_tasks()
        pass

    def num_tasks(self):
        return len(self.task_list)

    def num_executed_tasks(self):
        return len(self.executed_tasks)

    def num_outdated_tasks(self):
        return len(self.outdated_tasks)

    def num_active_tasks(self):
        return len(set(self.executed_tasks) & set(self.outdated_tasks))
    
    def get_task(self, task_id):
        return self.task_list[task_id]
    
    def is_executed(self, task_id):
        return task_id in self.executed_tasks

    def is_outdated(self, task_id):
        return task_id in self.outdated_tasks

    def is_active(self, task_id):
        return self.is_executed(task_id) and self.is_outdated(task_id)
    
    def check_missing_sources(self):
        """Check whether input files exist"""
        logger.debug('TaskGraph.check_missing_sources(): Collect sources that are not generated by any tasks')
        sources = set(sum([self.get_task(task_id).source for task_id in self.initial_tasks], []))
        logger.debug('TaskGraph.check_missing_sources(): check existence: %s', sources)
        missing_sources = []
        for source in sources:
            if not os.path.exists(source):
                missing_sources.append(source)
        logger.debug('TaskGraph.check_missing_sources(): missing sources: %s', missing_sources)
        return missing_sources
    
    def draw_dependencies(self):
        """Output a task graph in dot format"""
        out = StringIO()
        out.write('digraph dependencies {\n')
        out.write('  rankdir = LR;')
        out.write('  node [shape = box];\n')
        for task_id, task in enumerate(self.task_list):
            if self.is_active(task_id):
                style = 'solid'
            else:
                style = 'dashed'
            #out.write('  t{} [label="{}" style={}];\n'.format(task_id, ';'.join(task.rule), style))
            name = task.name.replace('\n', '\\n').replace('"', '\\"')
            out.write('  t{} [label="{}" style={}];\n'.format(task_id, name, style))
            prev_tasks = self.prev_tasks[task_id]
            for prev_task in prev_tasks:
                out.write('  t{} -> t{};\n'.format(prev_task, task_id))
        out.write('}\n')
        return out.getvalue().encode()
            
######################################################################

## parallel processing functions

class TaskTerminatedException(Exception):
    def __init__(self):
        pass
    
    def __str__(self):
        return 'Task terminated by SIGTERM'

def sigterm_handler(num, frame):
    raise TaskTerminatedException

class Worker(multiprocessing.Process):
    """Get a task from the queue and execute it in multiprocessing
    If it receives 'None', the worker terminates."""
    def __init__(self, worker_id, input_queue, output_queue, env=None):
        multiprocessing.Process.__init__(self)
        self.worker_id = worker_id
        self.input_queue = input_queue
        self.output_queue = output_queue
        if not (env is None or isinstance(env, dict)):
            raise ValueError('env must be a dict')
        self.env = env
        pass

    def run(self):
        os.setsid()  # disconnect from tty
        os.dup2(os.open(os.devnull, os.O_RDONLY), sys.stdin.fileno())  # disconnect stdin
        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore keyboard interrupt.  this is handled by main process
            signal.signal(signal.SIGTERM, sigterm_handler)  # raises exception for SIGTERM
            if self.env is not None:
                os.environ.update(self.env) # add environment
            while True:
                logger.debug('Worker %s waiting for task', self.name)
                (task_id, func) = self.input_queue.get()
                logger.debug('Worker %s got task', self.name)
                if func is None:
                    logger.debug('Worker %s terminates', self.name)
                    break  # terminate this process
                logger.debug('Worker %s runs task %s', self.name, task_id)
                try:
                    # execute the task
                    ret = func()
                except TaskTerminatedException as e:
                    # put the result as failure (to clean up the task execution)
                    logger.debug('Worker %s terminates task %s due to SIGTERM', self.name, task_id)
                    self.output_queue.put((self.worker_id, task_id, 1))
                    raise e
                except Exception as e:
                    # unexpected error raised
                    logger.error(coloring('red', 'Error in running task %s: %s'), task_id, e.message)
                    self.output_queue.put((self.worker_id, task_id, 1))
                    continue
                logger.debug('Worker %s finished task %s', self.name, task_id)
                self.output_queue.put((self.worker_id, task_id, ret))
                logger.debug('Worker %s has put the result of task %s', self.name, task_id)
        except TaskTerminatedException:
            # terminate the process
            logger.debug('Worker %s killing child processes', self.name)
            children = psutil.Process(self.pid).children(recursive=True)
            for child in children:
                child.terminate()
            logger.debug('Worker %s stops due to SIGTERM', self.name)
        return

class ExecCommand:
    def __init__(self, task_no, task, is_dry_run=False, touch=False, up_to_date=False):
        self.task_no = task_no
        self.task = task
        self.is_dry_run = is_dry_run  # do not run commands
        self.touch = touch            # run `touch` rather than executing commands
        self.up_to_date = up_to_date  # whether the targets are up-to-date
        pass

    def exec_touch(self, targets):
        for target in targets:
            try:
                os.utime(target, None)
            except:
                open(target, 'a').close()
        return 0
    
    def exec_command(self, rule):
        assert(isstr(rule))
        #return subprocess.call(rule, shell=True, close_fds=True, stdin=open(os.devnull), preexec_fn=os.setsid())
        #return subprocess.call(rule, shell=True, close_fds=True)
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # ignore keyboard interrupt.  this is handled by main process
        signal.signal(signal.SIGTERM, sigterm_handler)  # raises exception for SIGTERM
        try:
            logger.debug('ExecCommand:exec_command: %s', rule)
            proc = subprocess.Popen(rule, shell=True, close_fds=True, preexec_fn=os.setpgrp)
            ret = proc.wait()
            logger.debug('finished ExecCommand:exec_command with returncode=%s: %s', ret, rule)
        except OSError as e:
            # command cannot be executed
            logger.error(coloring('red', 'Command could not be executed: %s'), rule)
            return 1
        except TaskTerminatedException as e:
            logger.debug('ExecCommand terminates subprocess: %s', rule)
            os.killpg(proc.pid, signal.SIGTERM)
            return 1
        except Exception as e:
            logger.debug('ExecCommand: Unknown error raised')
            os.killpg(proc.pid, signal.SIGTERM)
            return 1
        return ret

    def __call__(self):
        if self.up_to_date:
            # does not execute the command because targets are up-to-date
            logger.info(coloring('blue', '%s [%s] targets up-to-date: ') + '%s', self.task_no, self.task.name, ', '.join(self.task.target))
            return 0
        elif self.is_dry_run:
            # dry-run mode: does not execute the command
            logger.info(coloring('green', '%s [%s] start: ') + '%s', self.task_no, self.task.name, self.task.show_rule())
            return 0
        elif self.touch:
            logger.info(coloring('green', '%s [%s] start: ') + '%s', self.task_no, self.task.name, self.task.show_rule())
            ret = self.exec_touch(self.task.target)
            return ret
        else:
            logger.info(coloring('green', '%s [%s] start: ') + '%s', self.task_no, self.task.name, self.task.show_rule())
            for rule in self.task.rule:
                ret = self.exec_command(rule)
                if ret != 0:
                    return ret
            logger.info(coloring('yellow', '%s [%s] done: ') + '%s', self.task_no, self.task.name, self.task.show_rule())
            return 0

class TaskFailedException(Exception):
    def __init__(self, task_id, ret):
        self.task_id = task_id
        self.ret = ret
        pass
    
    def __str__(self):
        return 'Task {} failed with status={}'.format(self.task_id, self.ret)

class Scheduler:
    def __init__(self, task_graph, dry_run=False, touch=False, keep_going=False, ignore_errors=False, num_jobs=1, environments=None, resources=None):
        if not isinstance(task_graph, TaskGraph):
            raise ValueError("task_graph must be an instance of TaskGraph")
        if environments is None:
            environments = [dict() for _ in range(num_jobs)]
        if not (isinstance(environments, list) and all([isinstance(x, dict) for x in environments])):
            raise ValueError("environments must be list of dict")
        if len(environments) != num_jobs:
            raise ValueError("length of environments must be equal to num_jobs")
        if resources is None:
            resources = [dict() for _ in range(num_jobs)]
        if not (isinstance(resources, list) and all([isinstance(x, dict) for x in resources])):
            raise ValueError("resources must be list of dict")
        if len(resources) != num_jobs:
            raise ValueError("length of resources must be equal to num_jobs")
        self.task_graph = task_graph
        self.dry_run = dry_run
        self.touch = touch
        self.keep_going = keep_going
        self.ignore_errors = ignore_errors
        self.num_jobs = num_jobs
        self.environments = environments
        self.resources = resources
        self.num_executed_tasks = 0
        self.num_failed_tasks = 0
        pass

    def __add_task(self, task_id, done_tasks, task_queue):
        """Add a new task on the task queue
        If the specified task is not outdated, the task is not executed and complete_task is called directly."""
        logger.debug('Scheduler.__add_task()')
        assert(self.task_graph.is_executed(task_id))  # task_id must be executed_task
        self.num_executed_tasks += 1
        task_no = '({}/{})'.format(self.num_executed_tasks, self.task_graph.num_executed_tasks())
        logger.debug('Scheduler.__add_task(): put task %s to the queue', task_id)
        command = ExecCommand(task_no, self.task_graph.task_list[task_id], is_dry_run=self.dry_run, touch=self.touch, up_to_date=not self.task_graph.is_outdated(task_id))
        task_queue.append((task_id, command))
        logger.debug('Scheduler.__add_task() done.  task queue size: %s', len(task_queue))
        pass

    def __complete_task(self, task_id, done_tasks, task_queue):
        """Complete the task and add the next tasks to the task queue"""
        logger.debug('Scheduler.__complete_task()')
        assert(self.task_graph.is_executed(task_id))  # task_id must be executed_task
        done_tasks.add(task_id)
        logger.debug('%s tasks have been finished so far', len(done_tasks))
        next_tasks = [tid for tid in self.task_graph.next_tasks[task_id] if self.task_graph.is_executed(tid)]
        logger.debug('After task %s, try adding the next %s tasks', task_id, len(next_tasks))
        tasks_to_add = []
        for next_task in next_tasks:
            prev_tasks = self.task_graph.prev_tasks[next_task]
            if all([t in done_tasks for t in prev_tasks]):
                logger.debug('Adding task %s after task %s since all the previous tasks done', next_task, task_id)
                tasks_to_add.append(next_task)
        for tid in tasks_to_add:
            self.__add_task(tid, done_tasks, task_queue)
        logger.debug('Scheduler.__complete_task() done')
        pass

    def __process_failed_task(self, task_id, ret):
        """Post-process failed task; show error message and remove target files"""
        logger.debug('Scheduler received task %s failure', task_id)
        self.num_failed_tasks += 1
        self.num_executed_tasks -= 1
        task = self.task_graph.task_list[task_id]
        logger.error(coloring('red', '***** [%s] failed (status=%s) *****: ') + '%s', task.name, ret, task.show_rule())
        logger.debug('Scheduler removes targets of task %s', task_id)
        self.__remove_targets(task_id)
        pass
    
    def __remove_targets(self, task_id):
        """Remove updated targets for the specified task
        Should be called when the task is failed"""
        # TODO: should remove only updated targets
        # at the moment, all targets are removed
        targets = self.task_graph.task_list[task_id].target
        if len(targets) > 0:
            logger.info(coloring('red', 'Removing targets: ') + '%s', ' '.join(targets))
            rename_suffix = '.failed-{}'.format(datetime.now().strftime("%Y%m%d%H%M%S"))
            for target in targets:
                if os.path.exists(target):
                    renamed = target + rename_suffix
                    os.rename(target, renamed)
                    logger.debug('Rename %s -> %s', target, renamed)
        pass

    def __assign_tasks(self, task_queue, worker_queue, input_queues):
        pending_tasks = []
        while len(worker_queue) > 0 and len(task_queue) > 0:
            task = task_queue.popleft()
            worker_id = None
            for i, id in enumerate(worker_queue):
                if self.task_graph.task_list[task[0]].resource_satisfied(self.resources[id]):
                    worker_index = i
                    worker_id = id
                    break
            if worker_id is None:
                # no worker satisfies resource requirement.  suspend this task.
                pending_tasks.append(task)
                continue
            # assign the task to worker_id
            worker_queue.pop(worker_index)
            logger.debug('Scheduler: assign task %s to worker %s', task[0], worker_id)
            input_queues[worker_id].put(task)
        # put back suspended tasks
        task_queue.extend(pending_tasks)

    def run(self):
        """Run tasks"""
        logger.debug('Scheduler.run()')
        missing_sources = self.task_graph.check_missing_sources()
        if len(missing_sources) > 0:
            for s in sorted(missing_sources):
                logger.error(coloring('red', 'Source not found: ') + '%s', s)
            self.num_failed_tasks = len(missing_sources)
            return  # Cannot run the tasks because some sources not found
        initial_tasks = self.task_graph.initial_tasks
        logger.debug('initial tasks: %s', initial_tasks)
        done_tasks = set()
        result_queue = multiprocessing.Queue()
        logger.debug('Scheduler creating %s workers', self.num_jobs)
        input_queues = [multiprocessing.Queue() for i in range(self.num_jobs)]
        worker_pool = [Worker(i, input_queues[i], result_queue, self.environments[i]) for i in range(self.num_jobs)]
        #worker_queue = deque(range(self.num_jobs))
        worker_queue = list(range(self.num_jobs))
        logger.debug('Scheduler adding %s initial tasks', len(initial_tasks))
        task_queue = deque()
        for task_id in initial_tasks:
            self.__add_task(task_id, done_tasks, task_queue)
        logger.debug('Scheduler starting %s workers', len(worker_pool))
        for worker in worker_pool:
            worker.start()
        try:
            signal.signal(signal.SIGTERM, sigterm_handler)
            # loop while the task is remaining or some workers are working
            while len(task_queue) > 0 or len(worker_queue) < len(worker_pool):
                # assign tasks in the queue to workers as far as possible
                logger.debug('Scheduler: assign tasks to workers: %s workers, %s tasks', len(worker_queue), len(task_queue))
                self.__assign_tasks(task_queue, worker_queue, input_queues)
                # retrieve task results and complete tasks
                worker_id, task_id, ret = result_queue.get()
                logger.debug('Scheduler recieved task %s result code %s from worker %s', task_id, ret, worker_id)
                if ret != 0 and not self.ignore_errors:
                    # task failed
                    self.__process_failed_task(task_id, ret)
                    if not self.keep_going:
                        logger.debug('Scheduler raise exception to stop scheduling')
                        raise TaskFailedException(task_id, ret)
                else:
                    # task succeeded
                    logger.debug('Scheduler: task %s finished successfully.  complete this task', task_id)
                    self.__complete_task(task_id, done_tasks, task_queue)
                logger.debug('Scheduler: worker %s is now available and added to the queue', worker_id)
                worker_queue.append(worker_id)
                logger.debug('Scheduler: done collecting results')
        except (TaskFailedException, TaskTerminatedException, KeyboardInterrupt) as e:
            # the scheduling is quit due to task failure, SIGTERM, or SIGINT
            logger.info(coloring('red', 'Terminating running tasks...'))
            logger.debug('Scheduler is stopping due to task failure, SIGTERM, or SIGINT')
            logger.debug('task_queue has %s items.  removing them.', len(task_queue))
            self.num_executed_tasks -= len(task_queue)
            task_queue.clear()
            logger.debug('Scheduler: terminate %s workers', len(worker_pool))
            for worker in worker_pool:
                worker.terminate()  # worker will terminate the current task with exit code 1
            if self.num_jobs > 1:
                logger.info(coloring('red', 'Waiting for running tasks to finish...'))
        logger.debug('Scheduler: all tasks finished.  Sending poison pill to the workers')
        for input_queue in input_queues:
            input_queue.put((0, None))  # poison pill
        logger.debug('Scheduler waiting for %s workers to terminate.', len(worker_pool))
        for worker in worker_pool:
            worker.join()
        logger.debug('Scheduler collecting remaining results from workers')
        while not result_queue.empty():
            # process finished tasks
            worker_id, task_id, ret = result_queue.get()
            if ret != 0:
                self.__process_failed_task(task_id, ret)
        logger.debug('Scheduler.run() done')
        pass

    def task_failed(self):
        return self.num_failed_tasks > 0
    
######################################################################

class Workflow:
    def set_options(self, num_jobs = None,
                    dry_run = None,
                    touch = None,
                    list_tasks = None,
                    dependency_graph = None,
                    keep_going = None,
                    ignore_errors = None,
                    always = None,
                    no_timestamp = None,
                    debug_level = None,
                    environment = None,
                    environments_distributed = None,
                    resources = None,
                    goal_targets = None):
        if num_jobs is not None:
            if not (isinstance(num_jobs, int) and num_jobs >= 1): raise ValueError("num_jobs must be positive int")
            self.num_jobs = num_jobs
        if dry_run is not None:
            if not isinstance(dry_run, bool): raise ValueError("dry_run must be bool")
            self.dry_run = dry_run
        if touch is not None:
            if not isinstance(touch, bool): raise ValueError("touch must be bool")
            self.touch = touch
        if list_tasks is not None:
            if not isinstance(list_tasks, bool): raise ValueError("list_tasks must be bool")
            self.list_tasks = list_tasks
        if dependency_graph is not None:
            if not isstr(dependency_graph): raise ValueError("dependency_graph must be string")
            self.dependency_graph = dependency_graph
        if keep_going is not None:
            if not isinstance(keep_going, bool): raise ValueError("keep_going must be bool")
            self.keep_going = keep_going
        if ignore_errors is not None:
            if not isinstance(ignore_errors, bool): raise ValueError("ignore_errors must be bool")
            self.ignore_errors = ignore_errors
        if always is not None:
            if not isinstance(always, bool): raise ValueError("always must be bool")
            self.always = always
        if no_timestamp is not None:
            if not isinstance(no_timestamp, bool): raise ValueError("no_timestamp must be bool")
            self.no_timestamp = no_timestamp
        if debug_level is not None:
            if debug_level not in [logging.DEBUG, logging.INFO, logging.ERROR]:
                raise ValueError("debug_level must be either of logging.DEBUG, logging.INFO, logging.ERROR")
            self.debug_level = debug_level
        if environment is not None:
            if not isinstance(environment, dict): raise ValueError("environment must be dict")
            self.environment = environment
        if environments_distributed is not None:
            if not (isinstance(environments_distributed, list) and all([isinstance(e, dict) for e in environments_distributed])):
                raise ValueError("environments_distributed must be list of dict")
            self.environments_distributed = environments_distributed
        if resources is not None:
            if not (isinstance(resources, list) and all([isinstance(r, dict) for r in resources])):
                raise ValueError("resources must be list of dict")
            self.resources = resources
        if goal_targets is not None:
            if not (isinstance(goal_targets, list) and all([isstr(x) for x in goal_targets])):
                raise ValueError("goal_targets must be list of strings")
            self.goal_targets = goal_targets
        pass
    
    def init_options(self,
                     num_jobs = 1,
                     dry_run = False,
                     touch = False,
                     list_tasks = False,
                     dependency_graph = None,
                     keep_going = False,
                     ignore_errors = False,
                     always = False,
                     no_timestamp = False,
                     debug_level = logging.INFO,
                     environment = None,
                     environments_distributed = None,
                     resources = None,
                     goal_targets = []):
        self.dependency_graph = None
        self.environment = None
        self.environments_distributed = None
        self.resources = None
        self.set_options(num_jobs = num_jobs,
                         dry_run = dry_run,
                         touch = touch,
                         list_tasks = list_tasks,
                         dependency_graph = dependency_graph,
                         keep_going = keep_going,
                         ignore_errors = ignore_errors,
                         always = always,
                         no_timestamp = no_timestamp,
                         debug_level = debug_level,
                         environment = environment,
                         environments_distributed = environments_distributed,
                         resources = resources,
                         goal_targets = goal_targets)
        pass
        
    def show_options(self):
        bools = ','.join([x[1] for x in
                          zip([self.dry_run, self.touch, self.list_tasks, self.keep_going, self.ignore_errors, self.always, self.no_timestamp],
                              ["dry_run", "touch", "list_tasks", "keep_going", "ignore_errors", "always", "no_timestamp"])
                          if x[0]])
        return """num_jobs={}, dependency_graph={}, debug_level={}, options: {}""".format(self.num_jobs, self.dependency_graph, self.debug_level, bools)
    
    def parse_args(self, args):
        argparser = argparse.ArgumentParser(description='A simple framework to run experiments')
        argparser.add_argument('-c', '--config', dest='config', type=str, default=None, help='Configuration file')
        argparser.add_argument('-j', '--jobs', dest='num_jobs', type=int, default=None, help='Number of jobs to run')
        argparser.add_argument('-n', '--dry-run', '--dryrun', dest='dry_run', action='store_true', default=None, help='Do not run commands (only print commands to be executed)')
        argparser.add_argument('-l', '--list-tasks', dest='list_tasks', action='store_true', default=None, help='Print the list of tasks')
        argparser.add_argument('-g', '--dependency-graph', dest='dependency_graph', type=str, default=None, help='Output dependency graph')
        argparser.add_argument('-k', '--keep-going', dest='keep_going', action='store_true', default=None, help='Run all tasks even when some tasks failed')
        argparser.add_argument('-S', '--no-keep-going', '--stop', dest='keep_going', action='store_false', default=None, help='Stop running tasks when some tasks failed (cancels "-k")')
        argparser.add_argument('-B', '--always', dest='always', action='store_true', default=None, help='Force running all commands')
        argparser.add_argument('-N', '--no-timestamp', dest='no_timestamp', action='store_true', default=None, help='Do not run commands if all sources/targets exist (do not check timestamp)')
        argparser.add_argument('-d', '--debug', dest='debug', action='store_true', default=None, help='Show debug messages')
        argparser.add_argument('-s', '--silent', '--quiet', dest='silent', action='store_true', default=None, help='Do not print commands to be run')
        argparser.add_argument('-t', '--touch', dest='touch', action='store_true', default=None, help='Touch target files rather than executing commands')
        argparser.add_argument('-i', '--ignore-errors', dest='ignore_errors', action='store_true', default=None, help='Ignore all errors in executed commands')
        argparser.add_argument('-e', '--environment', '--environment-overrides', dest='environment', action='store', default=None, help='Set environment variables (specify variable settings in JSON dictionary format)')
        argparser.add_argument('-E', '--environments-distributed', dest='environments_distributed', action='store', default=None, help='Set environment variables for each distributed worker (specify variable settings in the list of JSON dictionaries; the length of the list must be equal to the number of jobs')
        argparser.add_argument('-r', '--resources', dest='resources', action='store', default=None, help='Set resource specifications (specify resources in JSON dictionary format)')
        argparser.add_argument('target', nargs='*', default=None, help='Targets to be built (pattern match can be used; run all the tasks if not specified)')
        arguments = argparser.parse_args(args)
        logger.debug('options: %s', arguments)
        return arguments
    
    def __init__(self, args = sys.argv[1:], config = None, **options):
        """Initialize workflow
        - Three ways for initialization are supported
          - arguments of the constructor
          - configuration file
          - command-line arguments
        Latter overwrites former"""

        # initialize with constructor arguments
        self.init_options(**options)

        # parse command-line arguments
        if args is not None:
            arguments = self.parse_args(args)
            if arguments.config is not None:
                config = arguments.config

        # initialize with configuration file
        if config is not None:
            with open(config) as f:
                config_json = json.load(f)
                self.set_options(**config_json)

        # initialize with command-line arguments
        if arguments is not None:
            if arguments.debug:
                self.set_options(debug_level = logging.DEBUG)
            elif arguments.silent:
                self.set_options(debug_level = logging.ERROR)
            self.set_options(num_jobs = arguments.num_jobs,
                             dry_run = arguments.dry_run,
                             touch = arguments.touch,
                             list_tasks = arguments.list_tasks,
                             dependency_graph = arguments.dependency_graph,
                             keep_going = arguments.keep_going,
                             ignore_errors = arguments.ignore_errors,
                             always = arguments.always,
                             no_timestamp = arguments.no_timestamp,
                             goal_targets = arguments.target)
            if arguments.environment is not None:
                self.set_options(environment = json.loads(arguments.environment))
            if arguments.environments_distributed is not None:
                self.set_options(environments_distributed = json.loads(arguments.environments_distributed))
            if arguments.resources is not None:
                self.set_options(resources = json.loads(arguments.resources))
        self.task_list = []
        pass

    def __exists_equiv_task(self, task):
        """Check whether task_list already has the equivalent task (i.e. source and target are same)"""
        for t in self.task_list:
            if set(t.target) == set(task.target) and len(t.target) != 0:
                if set(t.source) == set(task.source):
                    if task.ignore_same_task and t.ignore_same_task:
                        logger.warn(coloring('purple', 'Ignored equivalent task: ') + '%s', task.name)
                        return True
                    else:
                        logger.error(coloring('red', 'Equivalent task found: ') + '%s', task.name)
                        raise ValueError('Equivalent task already defined: ' + task.name)
                else:
                    logger.error(coloring('red', 'Same target with different source found: ') + 'source=%s; target=%s', task.source, task.target)
                    raise ValueError('Tasks with same target but different source already defined: ' + task.name)
        return False
    
    def add(self, **args):
        task = Task(**args)
        if not self.__exists_equiv_task(task):
            self.task_list.append(task)

    def __call__(self, **args):
        self.add(**args)

    def __init_environments(self):
        "Set up environments to be set for each worker"
        if self.environment is None:
            envs = [dict() for _ in range(self.num_jobs)]
        else:
            envs = [self.environment.copy() for _ in range(self.num_jobs)]
        if self.environments_distributed is not None:
            if len(self.environments_distributed) != self.num_jobs:
                raise ValueError("length of environments_distributed must be equal to num_jobs: environments_distributed={}, num_jobs={}".format(len(self.environments_distributed), self.num_jobs))
            for env, new_env in zip(envs, self.environments_distributed):
                env.update(new_env)
        return envs

    def __init_resources(self):
        "Set up resources to be set for each worker"
        if self.resources is None:
            return [dict() for _ in range(self.num_jobs)]
        else:
            if len(self.resources) != self.num_jobs:
                raise ValueError("length of resources must be equal to num_jobs: resources={}, num_jobs={}".format(len(self.resources), self.num_jobs))
            return self.resources
        
    def __check_resources(self, resources):
        "Check whether all tasks have a required resource satisfiable with at least one worker"
        ret = True
        for task in self.task_list:
            resource_satisfied = False
            for available_resource in resources:
                if task.resource_satisfied(available_resource):
                    resource_satisfied = True
                    break
            if not resource_satisfied:
                logger.error(coloring('red', "Cannot find a worker to satisfy required resource: %s"), task.resource)
                ret = False
        return ret
    
    def show_task_list(self):
        for task in self.task_list:
            print(task.show_task())
        sources = set(sum([task.source for task in self.task_list], []))
        targets = set(sum([task.target for task in self.task_list], []))
        print("Sources")
        for source in sorted(list(sources)):
            print("  ", source)
        print()
        print("Targets")
        for target in sorted(list(targets)):
            print("  ", target)
        print()
        pass

    def run(self):
        logger.setLevel(self.debug_level)
        start = datetime.now()
        logger.debug('start run()')
        logger.debug(self.show_options())

        # Compute environments and resources
        environments = self.__init_environments()
        resources = self.__init_resources()
        assert(len(environments) == self.num_jobs)
        assert(len(resources) == self.num_jobs)
        if not self.__check_resources(resources):
            # resource satisfiability check failed
            return False
        
        # Run tasks
        logger.debug('create Scheduler')
        task_graph = TaskGraph(self.task_list, self.goal_targets, self.always, self.no_timestamp)
        scheduler = Scheduler(task_graph, dry_run=self.dry_run, touch=self.touch, keep_going=self.keep_going, ignore_errors=self.ignore_errors, num_jobs=self.num_jobs, environments=environments, resources=resources)
        if len(self.goal_targets) > 0:
            logger.info(coloring('blue', 'Targets: %s'), ' '.join(self.goal_targets))
        logger.info(coloring('blue', 'Total %s tasks'), task_graph.num_executed_tasks())
        logger.info(coloring('blue', 'Active %s tasks'), task_graph.num_active_tasks())

        if self.list_tasks:
            # print task list
            logger.info(coloring('green', 'Print the list of tasks'))
            self.show_task_list()
        elif self.dependency_graph is not None:
            # output dependency graph to a file
            logger.info(coloring('green', 'Output task dependency graph: %s'), self.dependency_graph)
            try:
                format = os.path.splitext(self.dependency_graph)[1][1:]  # extract format from file extension
                logger.debug('File format of dependency graph: %s', format)
                proc = subprocess.Popen(['dot', '-T', format, '-o', self.dependency_graph], stdin=subprocess.PIPE)
                proc.communicate(task_graph.draw_dependencies())
            except OSError:
                logger.error(coloring('red', "drawing task dependency graph failed: cannot execute 'dot'; install 'graphviz'"))
        else:
            # run the workflow
            logger.debug('run tasks')
            scheduler.run()

        if scheduler.task_failed():
            logger.info(coloring('red', '***** failed *****: %s failed, %s completed'), scheduler.num_failed_tasks, scheduler.num_executed_tasks)
        else:
            logger.info(coloring('blue', 'done: %s tasks completed'), scheduler.num_executed_tasks)
        logger.info(coloring('blue', 'Time: %s'), datetime.now() - start)
        if scheduler.task_failed():
            return False
        else:
            return True
            
######################################################################

def simple_task_example(exp):
    exp(depend='runexp.py', target='sample.txt', rule='sleep 1; echo "this is sample" > sample.txt')
    #exp(target='sample.txt', rule='sleep 1; echo "this is sample" > sample.txt', always=True)
    exp(source = 'sample.txt', target='target.txt', rule='sleep 2; cat sample.txt > target.txt')
    exp(source = 'sample.txt', target='target2.txt', rule='sleep 1; cat sample.txt > target2.txt')
    exp(source = 'target.txt', target='output.txt', rule='sleep 1; cat target.txt > output.txt')
    exp(source = 'output.txt', target='finaloutput.txt', rule='sleep 1; cat output.txt > finaloutput.txt')
    #exp(source = [], target='envsample.txt', rule='sleep 2; echo $ENV_SAMPLE > envsample.txt')
    pass

if __name__ == '__main__':
    try:
        exp = Workflow()
        logger.info('*** Running simple example of workflow ***')
        logger.info('See function `simple_task_example(exp)` for an example of defining a workflow')
        simple_task_example(exp)
        if not exp.run():
            sys.exit(1)
    except ValueError as e:
        logger.error(coloring('red', e.message))
        sys.exit(1)

