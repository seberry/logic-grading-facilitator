// Optional OCR failures do not prevent the TF from entering an answer manually.
export async function prepareAnswerReadings(tasks, request, onProgress = () => {}) {
  let completed = 0;
  const failed = [];
  for (const task of tasks) {
    onProgress({task, completed, failed: failed.length, total: tasks.length});
    try {
      const response = await request(task);
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        if (payload.code === 'vision_reading_format') {
          failed.push(task);
          continue;
        }
        throw new Error(`${task.problem_id}: ${payload.detail || response.status}`);
      }
      completed += 1;
    } catch (error) {
      error.completed = completed;
      throw error; // Stop for network/provider failure or changed configuration.
    }
  }
  return {completed, failed};
}
