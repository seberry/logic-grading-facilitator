import test from 'node:test';
import assert from 'node:assert/strict';
import {prepareAnswerReadings} from '../app/assets/fol_truth_batch.mjs';

const tasks = [1, 2, 3].map(unit_id => ({problem_id: 'Q1', unit_id}));
const response = (ok, payload) => ({ok, status: ok ? 200 : 502, json: async () => payload});

test('malformed optional readings do not stop later tasks and retries select only failures', async () => {
  const calls = [];
  const result = await prepareAnswerReadings(tasks, async task => {
    calls.push(task.unit_id);
    return task.unit_id === 2 ? response(false, {code: 'vision_reading_format', detail: 'Invalid JSON.'})
      : response(true, {status: task.unit_id === 1 ? 'cached' : 'proposed'});
  });
  assert.deepEqual(calls, [1, 2, 3]);
  assert.equal(result.completed, 2);
  assert.deepEqual(result.failed, [tasks[1]]);
  const retried = [];
  const retry = await prepareAnswerReadings(result.failed, async task => {
    retried.push(task.unit_id);
    return response(true, {status: 'proposed'});
  });
  assert.deepEqual(retried, [2]);
  assert.deepEqual(retry, {completed: 1, failed: []});
});

test('provider outages stop instead of repeatedly sending the rest of the batch', async () => {
  const calls = [];
  await assert.rejects(prepareAnswerReadings(tasks, async task => {
    calls.push(task.unit_id);
    return task.unit_id === 1 ? response(true, {status: 'cached'})
      : response(false, {detail: 'IU could not be reached.'});
  }), error => error.completed === 1 && /IU could not be reached/.test(error.message));
  assert.deepEqual(calls, [1, 2]);
});
