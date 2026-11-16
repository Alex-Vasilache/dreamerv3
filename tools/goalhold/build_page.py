"""Fill tools/goalhold/progress_page.html with the current figure, run and task tables.
Run after plot_progress.py. Edit RUNS / TASKS below as things change."""
import base64, datetime, sys
SRC = 'tools/goalhold/progress_page.html'
OUT = sys.argv[1]
RUNS = [
  ('h0410-h0433', 'batches 1-3: hold rules, thresholds, worker "done", controls, cheetah', '', 'V', 'ok', 'done'),
  ('h0434 / h0435 / h0437', 'cap 32 with real switch points: five reach / five "done" / four reach', '32', '', 'ok', 'done: 209 / 298 / 271'),
  ('h0438 / h0440 / h0441 / h0442', 'surprise (outliers) six seed 0 / five; surprise five seed 1; squared error x2', '8', '', 'ok', 'done: 0 / 304 / 296 / 293'),
  ('h0439', 'pin pad six, surprise (outliers), seed 1', '8', 'first pin pad six score', 'run', 'continuing to 6M: ~228 at 4.0M'),
  ('h0436', 'pin pad five, manager picks 1-8', '8', 'h0421', 'run', '1.2M: 0 (resumed after interruption)'),
  ('h0443', 'pin pad five, squared error x4', '8', 'h0442', 'run', 'started'),
  ('h0446 / h0447', 'pin pad five, real switch points: cap 32 seed 1 / cap 8', '', '', 'run', '2.5M: ~190 / 2.8M: ~296'),
  ('h0448 / h0449', 'real switch points, cap 32, on pin pad six / cheetah', '32', '', 'run', '1.7M: 0 / started'),
  ('h0444 / h0445', 'pin pad six, reach or 8 + transition-count bonus', '8', '', 'run', 'queued'),
  ('h0450-h0452', 'pin pad six, surprise (outliers), seeds 2-4, to 6M', '8', 'h0439', 'run', 'queued'),
]
TASKS = [
  ('Paper section', 'ok', 'done', 'Goal Flexibility: holding a goal until it is reached (15 pages, builds clean)'),
  ('Step-by-step tests of the whole path', 'ok', 'done', '24 checks on the real agent; older suites repaired (192 pass)'),
  ('Find the cause of the pin pad drop', 'ok', 'done', 'wrong bootstrap state for the cut last hold'),
  ('Fix and confirmation runs', 'run', 'confirmed', 'cap 8 finished: 332 / 350 on pin pad four (V 311)'),
  ('More bug-hunt angles', 'run', 'ongoing', 'second inconsistency found and fixed (replay switch points); it lifts cap 32'),
  ('Threshold, distance test, worker "done"', 'run', 'running', 'array 4823592'),
  ('Manager-chosen goal length', 'run', 'queued', 'array 4823592'),
  ('Surprise reward', 'run', 'first results', 'pin pad five 295-304 (277-293 without); one pin pad six seed solved'),
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
