"""Fill tools/goalhold/progress_page.html with the current figure, run and task tables.
Run after plot_progress.py. Edit RUNS / TASKS below as things change."""
import base64, datetime, sys
SRC = 'tools/goalhold/progress_page.html'
OUT = sys.argv[1]
RUNS = [
  ('h0410 / h0417', 'pin pad four, seeds 0 / 1', '8', 'h0377 (181), h0408 (323)', 'run', 'queued'),
  ('h0411 / h0416', 'pin pad four, seeds 0 / 1', '32', 'h0374 (237)', 'run', 'queued'),
  ('h0412', 'pin pad five', '32', 'h0375 (144), V 278', 'run', 'queued'),
  ('h0414', 'pin pad five', '8', 'h0381 (140)', 'run', 'queued'),
  ('h0413', 'pin pad six', '32', 'h0376 (0), V 1 of 5 seeds solved', 'run', 'queued'),
  ('h0415', 'pin pad six', '8', 'h0379 (0)', 'run', 'queued'),
]
TASKS = [
  ('Step-by-step tests of the whole path', 'ok', 'done', '24 checks on the real agent; older suites repaired (192 pass)'),
  ('Find the cause of the pin pad drop', 'ok', 'done', 'wrong bootstrap state for the cut last hold'),
  ('Fix and confirmation runs', 'run', 'running', 'array 4823338, h0410-h0417'),
  ('More bug-hunt angles', 'todo', 'next', 'trained checkpoint, real hold length in the environment, replay worker path'),
  ('Threshold, distance test, worker "done"', 'todo', 'next', 'after the fix is confirmed; cap 32'),
  ('Manager-chosen goal length', 'todo', 'next', 'rerun with the fix'),
  ('Surprise reward', 'todo', 'open', 'excess KL over a running baseline'),
]
s = open(SRC).read()
s = s.replace('__FIG__', base64.b64encode(open('/work/DoyaU/vasilache/work/meetings/goalhold/progress.png', 'rb').read()).decode())
s = s.replace('__DATE__', datetime.datetime.now().strftime('%d %b %Y, %H:%M JST'))
s = s.replace('__RUNS__', '\n'.join('<tr><td>%s</td><td class="r">%s</td><td>%s</td><td class="r">%s</td><td><span class="st %s">%s</span></td></tr>' % r for r in RUNS))
s = s.replace('__TASKS__', '\n'.join('<tr><td>%s</td><td><span class="st %s">%s</span></td><td class="r">%s</td></tr>' % t for t in TASKS))
open(OUT, 'w').write(s)
print('wrote', OUT, len(s))
