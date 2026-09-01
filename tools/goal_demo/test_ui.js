/* Headless checks on the page's non-DOM logic: the grid geometry, the code
 * keys the cache is built on, and the byte layout of a rendered frame.
 *
 *   NODE=$(readlink -f /proc/$(pgrep -u $USER -n node)/exe)
 *   $NODE tools/goal_demo/test_ui.js
 *
 * The script block is lifted straight out of index.html and run against a stub
 * DOM, so these test the code that actually ships rather than a copy of it.
 */
const fs = require('fs');
const path = require('path');
const vm = require('vm');

const html = fs.readFileSync(path.join(__dirname, 'index.html'), 'utf8');
const script = html.split('<script>')[1].split('</script>')[0]
    .replace('boot().catch(reportError);', '');

const stub = () => new Proxy(function () {}, {
  get: (t, k) => (k === 'classList' ? {toggle() {}, add() {}, remove() {}} :
                  k === 'style' ? {} :
                  k === 'dataset' ? {} :
                  k === 'children' ? [] :
                  k === 'childElementCount' ? 0 :
                  typeof k === 'string' ? stub() : undefined),
  set: () => true,
  apply: () => stub(),
});

const sandbox = {
  console,
  setTimeout, clearTimeout,
  fetch: () => Promise.reject(new Error('no network in tests')),
  ImageData: class {
    constructor(w, h) { this.width = w; this.height = h;
                        this.data = new Uint8ClampedArray(w * h * 4); }
  },
  document: {
    getElementById: stub,
    createElement: stub,
    querySelector: stub,
  },
};
vm.createContext(sandbox);
vm.runInContext(script + '\n;globalThis.__api = {' +
    'state, columns, keyOf, range, inRange, fmt, num, signed, valueFromX, ' +
    'rowCodes, missing, toImageData, jumpCapacities, drawDistance, ' +
    'splitDistance, jumpCode, codeAtDistance, pixelDiff, VALUE_KEYS};', sandbox);
const api = sandbox.__api;

let failures = 0;
function check(name, cond, got) {
  if (cond) { console.log(`  ok   ${name}`); return; }
  failures++;
  console.log(`  FAIL ${name}` + (got === undefined ? '' : ` -> ${JSON.stringify(got)}`));
}
function eq(name, a, b) { check(name, JSON.stringify(a) === JSON.stringify(b), a); }

api.state.extra = 4;
api.state.classes = 8;
api.state.blocks = 8;
const cols = api.columns();

console.log('grid geometry');
eq('columns span -4..11', cols, [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]);
check('0 and 7 are in the trained range',
    api.inRange(0) && api.inRange(7) && !api.inRange(-1) && !api.inRange(8));

console.log('pointer position -> class index');
// 16 cells of 20px starting at x=100, so cell i spans [100+20i, 120+20i).
const CELL = 20, X0 = 100;
const first = {left: X0, right: X0 + CELL};
const last = {left: X0 + 15 * CELL, right: X0 + 16 * CELL};
const at = (x, step = 1) => api.valueFromX(cols, first, last, x, step);
eq('centre of the first cell is -4', at(X0 + CELL / 2), -4);
eq('centre of the last cell is 11', at(X0 + 15 * CELL + CELL / 2), 11);
eq('centre of the class-0 cell is 0', at(X0 + 4 * CELL + CELL / 2), 0);
eq('centre of the class-7 cell is 7', at(X0 + 11 * CELL + CELL / 2), 7);
eq('dragging past the left end clamps', at(X0 - 500), -4);
eq('dragging past the right end clamps', at(X0 + 5000), 11);
eq('quarter steps land on quarters', at(X0 + 4 * CELL + CELL / 2 + CELL / 4, 0.25), 0.25);
check('every cell centre maps to its own column',
    cols.every((c, i) => at(X0 + i * CELL + CELL / 2) === c));

console.log('cache keys and prefetch sets');
eq('key is stable and per-block', api.keyOf([0, 1, 2, 3, 4, 5, 6, 7]),
    '0.000,1.000,2.000,3.000,4.000,5.000,6.000,7.000');
check('fractional values get distinct keys',
    api.keyOf([0.25, 0, 0, 0, 0, 0, 0, 0]) !== api.keyOf([0.5, 0, 0, 0, 0, 0, 0, 0]));
const self = {code: [3, 3, 3, 3, 3, 3, 3, 3], cache: new Map()};
eq('a snapped row is one code per column',
    api.rowCodes(self, 0, 1).length, 16);
eq('a fine row is one code per quarter step',
    api.rowCodes(self, 0, 0.25).length, 61);
check('row varies only the chosen block',
    api.rowCodes(self, 2, 1).every(c => c[0] === 3 && c[1] === 3 && c[3] === 3));
eq('fine row values stay clean',
    api.rowCodes(self, 0, 0.25).slice(0, 5).map(c => c[0]), [-4, -3.75, -3.5, -3.25, -3]);

const codes = api.rowCodes(self, 0, 1);
eq('nothing cached -> everything requested', api.missing(self, codes).length, 16);
self.cache.set(api.keyOf(codes[0]), 1);
eq('cached entries are dropped', api.missing(self, codes).length, 15);
eq('duplicates within one request are dropped',
    api.missing(self, [codes[5], codes[5], codes[6]]).length, 2);

console.log('frame decoding');
// Server sends (64, 64, 3) uint8, row-major; the page must not transpose it.
const SIZE = 64;
const raw = new Uint8Array(SIZE * SIZE * 3);
for (let i = 0; i < SIZE * SIZE; i++) {
  raw[i * 3] = i % 251; raw[i * 3 + 1] = (i * 7) % 251; raw[i * 3 + 2] = (i * 13) % 251;
}
const img = api.toImageData(raw, 0);
check('rgb channels land in order and alpha is opaque',
    [0, 1, 63, 64, 4095].every(i =>
        img.data[i * 4] === i % 251 &&
        img.data[i * 4 + 1] === (i * 7) % 251 &&
        img.data[i * 4 + 2] === (i * 13) % 251 &&
        img.data[i * 4 + 3] === 255));
const two = new Uint8Array(SIZE * SIZE * 3 * 2);
two.set(raw, SIZE * SIZE * 3);
check('a batched response is sliced at the right offset',
    api.toImageData(two, SIZE * SIZE * 3).data[4] === img.data[4]);

console.log('epsilon-greedy distant jump');
const C = 8, L = 8;
const dist = (a, b) => a.reduce((s, v, i) => s + Math.abs(v - b[i]), 0);

// d_max(c) = sum_l max(c_l, C-1-c_l): only an all-extremes code reaches L*(C-1).
const cap = c => api.jumpCapacities(c, C).reduce((a, b) => a + b, 0);
eq('all blocks at 0 reach the maximum', cap(new Array(L).fill(0)), L * (C - 1));
eq('all blocks at C-1 reach the maximum', cap(new Array(L).fill(C - 1)), L * (C - 1));
eq('all blocks mid-line reach the least', cap(new Array(L).fill(3)), L * 4);
eq('d_max is per-code, not constant', cap([0, 1, 2, 3, 4, 5, 6, 7]),
    7 + 6 + 5 + 4 + 4 + 5 + 6 + 7);

// p(d) = 2d / (d_max(d_max+1)) -> E[d] = (2/3)(d_max + 1/2).
let seed = 12345;
const rand = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
for (const dmax of [44, 56]) {
  const N = 40000;
  let sum = 0, lo = Infinity, hi = -Infinity;
  const hist = new Array(dmax + 1).fill(0);
  for (let i = 0; i < N; i++) {
    const d = api.drawDistance(dmax, rand);
    sum += d; hist[d]++; lo = Math.min(lo, d); hi = Math.max(hi, d);
  }
  const want = (2 / 3) * (dmax + 0.5);
  check(`d_max ${dmax}: mean d ${(sum / N).toFixed(2)} vs ${want.toFixed(2)}`,
      Math.abs(sum / N - want) < 0.5);
  check(`d_max ${dmax}: d stays in 1..d_max`, lo >= 1 && hi <= dmax, [lo, hi]);
  // p(d) grows linearly, so the top half must carry three times the bottom.
  const half = Math.floor(dmax / 2);
  const bottom = hist.slice(1, half + 1).reduce((a, b) => a + b, 0);
  const top = hist.slice(half + 1).reduce((a, b) => a + b, 0);
  check(`d_max ${dmax}: large jumps dominate (top/bottom ${(top / bottom).toFixed(2)})`,
      top / bottom > 2.5);
}

// The whole point: the sampled code is at EXACTLY the drawn distance.
let exact = 0, offLine = 0, moved = 0, trials = 3000, sumd = 0;
for (let i = 0; i < trials; i++) {
  const base = Array.from({length: L}, () => Math.floor(rand() * C));
  const {code, d, dmax} = api.jumpCode(base, C, rand);
  if (dist(code, base) === d) exact++;
  if (code.some(v => v < 0 || v > C - 1 || !Number.isInteger(v))) offLine++;
  if (d > 0) moved++;
  sumd += d;
}
eq('every jump lands at exactly the drawn distance', exact, trials);
eq('no jump leaves the class range', offLine, 0);
eq('every jump moves', moved, trials);
check(`mean drawn d over random codes is ~30 (got ${(sumd / trials).toFixed(1)})`,
    Math.abs(sumd / trials - 30) < 4);

// The paper's claim: over uniformly drawn codes d_max averages 44, not 56.
// E[max(c, C-1-c)] over c in 0..7 is (7+6+5+4+4+5+6+7)/8 = 5.5, times L = 44.
let sumcap = 0;
for (let i = 0; i < 20000; i++) {
  sumcap += cap(Array.from({length: L}, () => Math.floor(rand() * C)));
}
check(`d_max averages 44 over random codes (got ${(sumcap / 20000).toFixed(1)})`,
    Math.abs(sumcap / 20000 - 44) < 0.5);

// A code with no room to move must not pretend it jumped.
eq('a zero-capacity code stays put', api.jumpCode([0], 1, rand).d, 0);

console.log('a new state at exactly the distance you asked for');
// The point of the button: whatever d you type, the code lands there.
let hit = 0, off = 0, over = 0;
for (let i = 0; i < 2000; i++) {
  const base = Array.from({length: L}, () => Math.floor(rand() * C));
  const want = Math.floor(rand() * 40);
  const {code, d} = api.codeAtDistance(base, C, want, rand);
  if (dist(code, base) === want) hit++;
  if (d !== want) off++;
  if (code.some(v => v < 0 || v > C - 1 || !Number.isInteger(v))) over++;
}
eq('every generated code sits at exactly the requested distance', hit, 2000);
eq('the reported d is the requested one', off, 0);
eq('no generated code leaves the class range', over, 0);
eq('distance 0 gives the code back', api.codeAtDistance([1, 2, 3], C, 0, rand).code,
    [1, 2, 3]);
// Beyond d_max there is nowhere left to go, and it must say so rather than lie.
const maxed = api.codeAtDistance([3, 3], C, 99, rand);
eq('an unreachable distance falls back to d_max', [maxed.d, maxed.dmax], [8, 8]);
eq('a fractional code is rounded before it is counted',
    api.codeAtDistance([2.4, 5.6], C, 0, rand).code, [2, 6]);
// The jump is now this function with p(d) in front, so it must not have drifted.
seed = 999;
const j = api.jumpCode([0, 1, 2, 3, 4, 5, 6, 7], C, rand);
eq('the jump still lands at its own drawn distance',
    dist(j.code, [0, 1, 2, 3, 4, 5, 6, 7]), j.d);

console.log('reading out critic values');
eq('the three critics are shown separately, never pooled',
    api.VALUE_KEYS.filter(v => v.head.endsWith('_val')).map(v => v.head),
    ['mgr_extr_val', 'mgr_expl_val', 'wkr_goal_val']);
// Critic outputs span ~0.4 to ~800, so precision has to follow magnitude.
eq('a task return keeps one decimal', api.num(227.3096), '227.3');
eq('an exploration value keeps three', api.num(0.4134), '0.413');
eq('a change carries its sign', [api.signed(-16.72), api.signed(3)],
    ['−16.72', '+3.000']);

// Picture difference: mean |delta| per channel, alpha ignored.
const black = new sandbox.ImageData(SIZE, SIZE);
const white = new sandbox.ImageData(SIZE, SIZE);
for (let i = 0; i < SIZE * SIZE; i++) {
  for (let c = 0; c < 3; c++) white.data[i * 4 + c] = 255;
  black.data[i * 4 + 3] = white.data[i * 4 + 3] = 255;
}
eq('identical frames differ by nothing', api.pixelDiff(black, black), 0);
eq('black against white is the full range', api.pixelDiff(black, white), 255);

console.log(failures ? `\n${failures} failing` : '\nall ui checks passed');
process.exit(failures ? 1 : 0);
