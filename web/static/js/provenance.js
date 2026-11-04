/* LLM Forensics - Provenance */

/** The job endpoint of this page (POST to start, GET <id> to poll). */
const JOB_URL = '/api/forensics/provenance';

/** The input panels folded away while a run is in progress (and behind the finished summary). */
const INPUT_PANELS = ['input-panel'];

/** The model picker of the suspect, created once the DOM is ready. */
const pickers = {};

/**
 * Show the search options only when the Hub search is on.
 */
function handleSearchToggle() {
    document.getElementById('search-options').style.display =
        document.getElementById('search-toggle').checked ? '' : 'none';
}

/**
 * Handle the run button click: collapse the inputs, run, then show the verdict summary.
 */
async function handleRun() {
    if (await ensureAllDetected([pickers.suspect])) {
        const payload = {
            suspect: pickers.suspect.value(),
            candidates: document.getElementById('candidates-input').value,
            search: document.getElementById('search-toggle').checked,
            search_text: document.getElementById('search-text').value.trim(),
            include_derived: document.getElementById('derived-toggle').checked,
            layers: document.getElementById('layers-input').value,
            top_k: document.getElementById('topk-input').value,
        };
        const result = await runJob(JOB_URL, JOB_URL, payload, 'Finding the provenance', null, null,
            (collapsed) => collapsePanels(INPUT_PANELS, collapsed));
        if (result) renderToolResult(result);   // success: keep the inputs collapsed behind the summary
    }
}

/**
 * Reattach to this page's running or just-finished search after a navigation.
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
    pickers.suspect = new ModelPicker('suspect');
    new HubAutocomplete(document.getElementById('search-text'), document.getElementById('search-text-dropdown'));
    document.getElementById('run-btn').addEventListener('click', handleRun);
    initJobReattach('provenance', reattach);
    document.getElementById('search-toggle').addEventListener('change', handleSearchToggle);
    document.getElementById('candidates-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) handleRun();
    });
    ['search-text', 'layers-input', 'topk-input'].forEach((id) => {
        document.getElementById(id).addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handleRun();
        });
    });
});
