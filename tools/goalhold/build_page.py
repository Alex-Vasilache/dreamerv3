"""Fill tools/goalhold/progress_page.html with the current figure, run and task tables.
Run after plot_progress.py. Edit RUNS / TASKS below as things change."""
import base64, datetime, sys
SRC = 'tools/goalhold/progress_page.html'
OUT = sys.argv[1]
RUNS = [
  ('h0410 / h0417', 'pin pad four, reach or 8', '8', 'V 311, before fix 181', 'ok', 'done: 332 / 350'),
  ('h0414', 'pin pad five, reach or 8', '8', 'V 278, before fix 140', 'ok', 'done: 277, first score at 0.30M'),
  ('h0411 / h0416', 'pin pad four, reach or 32', '32', 'before fix 237', 'run', 'running, ~3.2M'),
  ('h0412', 'pin pad five, reach or 32', '32', 'before fix 144', 'run', 'running'),
  ('h0413 / h0415', 'pin pad six, cap 32 / 8', '', 'V solves 1 of 5 seeds', 'run', 'running, 0 so far'),
  ('h0418 / h0419', 'pin pad five, threshold 0.7 / 0.9', '32', 'h0412', 'run', 'started'),
  ('h0420', 'pin pad five, worker "done"', '32', 'h0412, h0414', 'run', 'started'),
  ('h0421 / h0425', 'pin pad five / four, manager picks length', '1-32', 'h0414, h0410', 'run', 'queued'),
  ('h0422 / h0423', 'pin pad five / six, surprise reward', '8', 'h0414, h0415', 'run', 'queued'),
  ('h0424', 'pin pad five, reach or 8, seed 1', '8', 'h0414', 'run', 'queued'),
]
TASKS = [
  ('Step-by-step tests of the whole path', 'ok', 'done', '24 checks on the real agent; older suites repaired (192 pass)'),
  ('Find the cause of the pin pad drop', 'ok', 'done', 'wrong bootstrap state for the cut last hold'),
  ('Fix and confirmation runs', 'run', 'confirmed', 'cap 8 finished: 332 / 350 on pin pad four (V 311)'),
  ('More bug-hunt angles', 'todo', 'next', 'trained checkpoint, real hold length in the environment, replay worker path'),
  ('Threshold, distance test, worker "done"', 'run', 'running', 'array 4823592'),
  ('Manager-chosen goal length', 'run', 'queued', 'array 4823592'),
  ('Surprise reward', 'run', 'queued', 'array 4823592'),
  ('Real hold length in the environment', 'run', 'ok', 'working: real holds 6-18 steps in the first batch-2 runs'),
]
s = open(SRC).read()
s = s.replace('__FIG__', base64.b64encode(open('/work/DoyaU/vasilache/work/meetings/goalhold/progress.png', 'rb').read()).decode())
s = s.replace('__DATE__', datetime.datetime.now().strftime('%d %b %Y, %H:%M JST'))
s = s.replace('__RUNS__', '\n'.join('<tr><td>%s</td><td class="r">%s</td><td>%s</td><td class="r">%s</td><td><span class="st %s">%s</span></td></tr>' % r for r in RUNS))
s = s.replace('__TASKS__', '\n'.join('<tr><td>%s</td><td><span class="st %s">%s</span></td><td class="r">%s</td></tr>' % t for t in TASKS))
open(OUT, 'w').write(s)
print('wrote', OUT, len(s))
