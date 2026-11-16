"""Fill tools/goalhold/progress_page.html with the current figure, run and task tables.
Run after plot_progress.py. Edit RUNS / TASKS below as things change."""
import base64, datetime, sys
SRC = 'tools/goalhold/progress_page.html'
OUT = sys.argv[1]
RUNS = [
  ('h0410-h0433', 'batches 1-3: hold rules, thresholds, worker "done", controls, cheetah', '', 'V', 'ok', 'done'),
  ('h0434-h0437', 'cap 32 with real switch points; manager picks 1-8', '', '', 'ok', 'done'),
  ('h0438-h0443', 'surprise (first and outliers-only version); squared-error reached test', '8', '', 'ok', 'done'),
  ('h0439', 'pin pad six, surprise (outliers), seed 1, continued to 6M', '8', 'first pin pad six score', 'ok', 'done: 215-225 from 4M'),
  ('h0446 / h0447', 'pin pad five, real switch points: cap 32 seed 1 / cap 8', '', '', 'ok', 'done: 214 / 292'),
  ('h0449', 'cheetah, reach or 32 with real switch points', '32', 'reach or 8: 616', 'run', '3.1M: ~340'),
  ('h0444 / h0445', 'pin pad six, reach or 8 + transition-count bonus', '8', '', 'run', '3.9M / 1.8M: 0'),
  ('h0450-h0452', 'pin pad six, surprise (outliers), seeds 2-4, to 6M', '8', 'h0439', 'run', '2.1-2.9M: 0'),
  ('h0453', 'pin pad five, "done" + real switch points, seed 1', '32', 'h0435 (298)', 'run', 'waiting for a GPU'),
]
TASKS = [
  ('Paper section', 'ok', 'done', 'Goal Flexibility: holding a goal until it is reached (15 pages, builds clean)'),
  ('Step-by-step tests of the whole path', 'ok', 'done', '24 checks on the real agent; older suites repaired (192 pass)'),
  ('Find the cause of the pin pad drop', 'ok', 'done', 'wrong bootstrap state for the cut last hold'),
  ('Fix and confirmation runs', 'run', 'confirmed', 'cap 8 finished: 332 / 350 on pin pad four (V 311)'),
  ('More bug-hunt angles', 'run', 'ongoing', 'second inconsistency found and fixed (replay switch points); it lifts cap 32'),
  ('Threshold, distance test, worker "done"', 'ok', 'done', 'see the table above'),
  ('Manager-chosen goal length', 'ok', 'done', 'fails at 1-32 and at 1-8'),
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
