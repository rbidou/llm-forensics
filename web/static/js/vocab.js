/* LLM Forensics - Vocabulary & Embeddings */

/** The job endpoint of this page (POST to start, GET <id> to poll). */
const JOB_URL = '/api/forensics/vocab';

/** The input panels folded away while a run is in progress (and behind the finished summary). */
const INPUT_PANELS = ['input-panel'];

/** The model pickers of the page, created once the DOM is ready. */
const pickers = {};

/**
 * Handle the compare button click: collapse the inputs, run, then show the risk summary.
 */
async function handleRun() {
    if (await ensureAllDetected([pickers.base, pickers.tuned])) {
        const payload = {
            base: pickers.base.value(),
            tuned: pickers.tuned.value(),
            top_rows: document.getElementById('rows-input').value,
            new_tokens: document.getElementById('new-input').value,
        };
        const result = await runJob(JOB_URL, JOB_URL, payload, 'Comparing vocabulary and embeddings', null, null,
            (collapsed) => collapsePanels(INPUT_PANELS, collapsed));
        if (result) renderToolResult(result);
    }
}

/**
 * Reattach to this page's running or just-finished comparison after a navigation.
 * @param {object} job - the current-job payload
 */
async function reattach(job) {
    collapsePanels(INPUT_PANELS, true);
    const result = await followJob(JOB_URL, job.job_id, job.label);
    if (result) {
        renderToolResult(result);
    } else {
        collapsePanels(INPUT_PANELS, false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    pickers.base = new ModelPicker('base');
    pickers.tuned = new ModelPicker('tuned');
    document.getElementById('run-btn').addEventListener('click', handleRun);
    initJobReattach('vocab', reattach);
    ['rows-input', 'new-input'].forEach((id) => {
        document.getElementById(id).addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handleRun();
        });
    });
});
