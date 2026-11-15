"""Fill tools/goalhold/progress_page.html with the current figure, run and task tables.
Run after plot_progress.py. Edit RUNS / TASKS below as things change."""
import base64, datetime, sys
SRC = 'tools/goalhold/progress_page.html'
OUT = sys.argv[1]
RUNS = [
  ('h0410-h0417', 'batch 1: reach or 8 / 32 on pin pads four, five, six', '8, 32', 'V', 'ok', 'done'),
  ('h0418', 'pin pad five, threshold 0.7', '32', 'h0412', 'run', '3M: 220'),
  ('h0419', 'pin pad five, threshold 0.9', '32', 'h0412', 'ok', 'stopped at 2.2M: 0'),
  ('h0420', 'pin pad five, worker "done"', '32', 'h0412, h0414', 'run', '3M: 290'),
  ('h0421 / h0425', 'pin pad five / four, manager picks 1-32', '32', 'h0414, h0410', 'ok', 'stopped: not learning'),
  ('h0422', 'pin pad five, reach or 8 + surprise', '8', 'h0414', 'run', '3M: 296'),
  ('h0423', 'pin pad six, reach or 8 + surprise', '8', 'h0415', 'run', '2.5M: 0'),
  ('h0424', 'pin pad five, reach or 8, seed 1', '8', 'h0414', 'run', '2.5M: 278'),
  ('h0426 / h0427 / h0433', 'worker "done" on pin pad four / six / five seed 1', '32', '', 'run', 'started'),
  ('h0428', 'pin pad five, fixed 4 steps (control)', '4', 'h0414', 'run', 'queued'),
  ('h0429-h0432', 'pin pad six surprise seed 1, cheetah, cap 16, done + surprise', '', '', 'run', 'queued'),
  ('h0434-h0437', 'real switch points in replay (cap 32); manager picks 1-8', '', '', 'run', 'queued'),
]
TASKS = [
  ('Step-by-step tests of the whole path', 'ok', 'done', '24 checks on the real agent; older suites repaired (192 pass)'),
  ('Find the cause of the pin pad drop', 'ok', 'done', 'wrong bootstrap state for the cut last hold'),
  ('Fix and confirmation runs', 'run', 'confirmed', 'cap 8 finished: 332 / 350 on pin pad four (V 311)'),
  ('More bug-hunt angles', 'run', 'ongoing', 'found: replay trained on switch points the agent never used; fix queued'),
  ('Threshold, distance test, worker "done"', 'run', 'running', 'array 4823592'),
  ('Manager-chosen goal length', 'run', 'queued', 'array 4823592'),
  ('Surprise reward', 'run', 'queued', 'array 4823592'),
  ('Real hold length in the environment', 'run', 'ok', 'working: real holds 6-18 steps in the first batch-2 runs'),
]
s = open(SRC).read()
s = s.replace('__FIG__', base64.b64encode(open('/work/DoyaU/vasilache/work/meetings/goalhold/progress.png', 'rb').read()).decode())
s = s.replace('__FIG2__', base64.b64encode(open('/work/DoyaU/vasilache/work/meetings/goalhold/progress2.png', 'rb').read()).decode())
s = s.replace('__DATE__', datetime.datetime.now().strftime('%d %b %Y, %H:%M JST'))
s = s.replace('__RUNS__', '\n'.join('<tr><td>%s</td><td class="r">%s</td><td>%s</td><td class="r">%s</td><td><span class="st %s">%s</span></td></tr>' % r for r in RUNS))
s = s.replace('__TASKS__', '\n'.join('<tr><td>%s</td><td><span class="st %s">%s</span></td><td class="r">%s</td></tr>' % t for t in TASKS))
open(OUT, 'w').write(s)
print('wrote', OUT, len(s))
