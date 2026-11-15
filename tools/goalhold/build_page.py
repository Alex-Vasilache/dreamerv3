"""Fill tools/goalhold/progress_page.html with the current figure, run and task tables.
Run after plot_progress.py. Edit RUNS / TASKS below as things change."""
import base64, datetime, sys
SRC = 'tools/goalhold/progress_page.html'
OUT = sys.argv[1]
RUNS = [
  ('h0410-h0417', 'batch 1: reach or 8 / 32 on pin pads four, five, six', '8, 32', 'V', 'ok', 'done'),
  ('h0418-h0425', 'batch 2: thresholds, worker "done", manager picks 1-32, surprise', '', '', 'ok', 'done'),
  ('h0426', 'pin pad four, worker "done"', '32', 'h0410 (332)', 'run', '3.1M: 288'),
  ('h0427 / h0428', 'pin pad six "done" / fixed-4 control', '', '', 'ok', 'stopped at 0'),
  ('h0429', 'pin pad six, surprise, seed 1', '8', '', 'run', '1.9M: 0'),
  ('h0430', 'cheetah, reach or 8', '8', 'V 536 late', 'run', '1.3M: ~500'),
  ('h0431', 'pin pad five, reach or 16', '16', 'h0414, h0412', 'run', '1.4M: 233'),
  ('h0432', 'pin pad five, "done" + surprise', '32', 'h0420, h0422', 'run', '1.2M: 248'),
  ('h0433', 'pin pad five, "done", seed 1', '32', 'h0420', 'run', '1.0M: 264'),
  ('h0434-h0437', 'real switch points in replay (cap 32); manager picks 1-8', '', '', 'run', 'starting'),
  ('h0438-h0441', 'surprise that pays only outliers; second surprise seed', '8', '', 'run', 'queued'),
  ('h0442-h0443', 'reached by squared error', '8', 'h0414', 'run', 'queued'),
  ('h0444-h0445', 'pin pad six, reach or 8 + transition-count bonus', '8', '', 'run', 'queued'),
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
