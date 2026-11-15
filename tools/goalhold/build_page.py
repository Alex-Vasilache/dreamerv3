"""Fill tools/goalhold/progress_page.html with the current figure, run and task tables.
Run after plot_progress.py. Edit RUNS / TASKS below as things change."""
import base64, datetime, sys
SRC = 'tools/goalhold/progress_page.html'
OUT = sys.argv[1]
RUNS = [
  ('h0410-h0417', 'batch 1: reach or 8 / 32 on pin pads four, five, six', '8, 32', 'V', 'ok', 'done'),
  ('h0418-h0425', 'batch 2: thresholds, worker "done", manager picks 1-32, surprise', '', '', 'ok', 'done'),
  ('h0426-h0433', 'batch 3: "done" follow-ups, fixed-4 control, cap 16, cheetah (616)', '', '', 'ok', 'done'),
  ('h0434 / h0435', 'pin pad five, cap 32 with real switch points: reach / "done"', '32', 'h0412, h0420', 'ok', 'done: 209 / 298'),
  ('h0437', 'pin pad four, reach or 32 with real switch points', '32', 'h0411 (284)', 'run', '3.7M: ~280'),
  ('h0436', 'pin pad five, manager picks 1-8', '8', 'h0421', 'run', 'interrupted by the cluster at 0.9M, waiting'),
  ('h0438 / h0439', 'pin pad six, surprise paying only outliers, seeds 0 / 1', '8', '', 'run', '2.0M: 0'),
  ('h0440', 'pin pad five, surprise paying only outliers', '8', 'h0422', 'run', '2.0M: ~290'),
  ('h0441', 'pin pad five, reach or 8 + surprise, seed 1', '8', 'h0422', 'run', '2.0M: ~270'),
  ('h0442 / h0443', 'pin pad five, reached by squared error (x2 / x4)', '8', 'h0414', 'run', 'x2 at 2.1M: ~280; x4 queued'),
  ('h0446 / h0447', 'pin pad five, real switch points: cap 32 seed 1 / cap 8', '', '', 'run', 'started'),
  ('h0448 / h0449', 'real switch points on pin pad six / cheetah', '32', '', 'run', 'queued'),
  ('h0444 / h0445', 'pin pad six, reach or 8 + transition-count bonus', '8', '', 'run', 'queued'),
]
TASKS = [
  ('Paper section', 'ok', 'done', 'Goal Flexibility: holding a goal until it is reached (15 pages, builds clean)'),
  ('Step-by-step tests of the whole path', 'ok', 'done', '24 checks on the real agent; older suites repaired (192 pass)'),
  ('Find the cause of the pin pad drop', 'ok', 'done', 'wrong bootstrap state for the cut last hold'),
  ('Fix and confirmation runs', 'run', 'confirmed', 'cap 8 finished: 332 / 350 on pin pad four (V 311)'),
  ('More bug-hunt angles', 'run', 'ongoing', 'second inconsistency found and fixed (replay switch points); it lifts cap 32'),
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
