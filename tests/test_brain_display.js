const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const context = vm.createContext({});
vm.runInContext(fs.readFileSync(path.join(__dirname, '../flytrader/static/brain.js'), 'utf8'), context);
const {activityScale, activityChanges, activityPulse} = context;
const frame = (step, values, mint = 'COIN') => ({step, values, mint});

test('auto contrast reveals small states without changing their recorded values', () => {
  const values = [-.003, -.002, .002, .001, 0];
  const original = values.slice();
  assert.ok(activityScale(values, 'auto') < .01);
  assert.deepEqual(values, original);
  assert.equal(activityScale(values, '0.1'), .1);
  assert.equal(activityScale(values, '1'), 1);
});

test('one strong outlier does not wash out the other cells', () => {
  assert.equal(activityScale([...Array(999).fill(.002), 1], 'auto'), .002);
});

test('zero states and rounding-scale noise are not amplified into strong colors', () => {
  assert.equal(activityScale([], 'auto'), .001);
  assert.equal(activityScale(Array(2000).fill(0), 'auto'), .001);
  assert.equal(activityScale([-.000001, .000001], 'auto'), .001);
});

test('change highlights use actual signed differences ranked by magnitude', () => {
  const previous = frame(10, [0, -.002, .003, .001]);
  const current = frame(11, [.000001, -.002, -.004, .006]);
  const result = activityChanges(previous, current);
  assert.equal(result.count, 3);
  assert.equal(result.comparedStep, 10);
  assert.deepEqual(Array.from(result.changes, change => change.index), [2, 3, 0]);
  assert.equal(result.changes[0].delta, -.007);
  assert.equal(current.values[2], -.004);
});

test('identical observations do not produce activity highlights', () => {
  const result = activityChanges(frame(10, [.002, -.001]), frame(11, [.002, -.001]));
  assert.equal(result.count, 0);
  assert.equal(result.changes.length, 0);
});

test('first frames, another coin, backwards playback, and shape changes have no fabricated comparison', () => {
  const current = frame(10, [0, .001]);
  for (const previous of [null, frame(9, [0, .001], 'OTHER'), frame(11, [0, .001]), frame(10, [0, .001]), frame(9, [0])]) {
    const result = activityChanges(previous, current);
    assert.equal(result.count, null);
    assert.equal(result.changes.length, 0);
  }
});

test('skipped live updates identify the actual observation used for comparison', () => {
  const result = activityChanges(frame(20, [0, 0]), frame(25, [0, .002]));
  assert.equal(result.count, 1);
  assert.equal(result.comparedStep, 20);
});

test('measured-change animation remains visible between ten-second price updates', () => {
  assert.notEqual(activityPulse(3000, true), activityPulse(3600, true));
  assert.ok(activityPulse(9000, true) > 0);
  assert.equal(activityPulse(30000, true), 0);
  assert.equal(activityPulse(3000, false), 0);
});
