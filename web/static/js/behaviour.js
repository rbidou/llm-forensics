/* LLM Forensics - Behaviour Probe
   Batch mode uploads a prompts file and runs one probe job. Interactive mode
   loads the pair once (a job), then compares it on one prompt per request; the
   server keeps the pair until it is unloaded or another one is loaded. */

/** The endpoints of this page. */
const BATCH_URL = '/api/forensics/behaviour';
const POLL_URL = '/api/forensics/behaviour';
const SESSION_URL = '/api/forensics/behaviour/session';
const PROMPT_URL = '/api/forensics/behaviour/session/prompt';
const SAVE_URL = '/api/forensics/behaviour/session/save';

/** The upload limit, also enforced by the server. */
const MAX_FILE_SIZE = 2 * 1024 * 1024;

/** The prompt file types the probe reads. */
const ALLOWED_SUFFIXES = ['.txt', '.json', '.jsonl'];

/** The input panels folded away while the batch probe runs (and behind the finished summary). */
const INPUT_PANELS = ['models-panel', 'batch-panel'];

/** Rows shown in the tables (the report keeps everything). */
const HOTTEST_SHOWN = 15;
const SHIFTS_SHOWN = 15;
const PREVIEW_CHARS = 120;

/** The model pickers of the page, created once the DOM is ready. */
const pickers = {};

/** The prompts file chosen for the batch probe. */
let selectedFile = null;

/* ---------- Rendering helpers ---------- */

/**
 * A shortened prompt for a table cell.
 * @param {string} text
 * @returns {string}
 */
function preview(text) {
    const flat = String(text || '').replace(/\s+/g, ' ').trim();
    return flat.length > PREVIEW_CHARS ? `${flat.slice(0, PREVIEW_CHARS)}…` : flat;
}

/**
 * Token chips for promoted or suppressed tokens.
 * @param {object[]} shifts - {token_id, text, shift}
 * @param {string} extraClass - 'up' or 'down'
 * @returns {string} HTML
 */
function shiftChips(shifts, extraClass) {
    const chips = (shifts || []).slice(0, SHIFTS_SHOWN)
        .map((item) => tokenChip(item.text, extraClass, `id ${item.token_id}, Δ log-prob ${formatNumber(item.shift)}`));
    return chips.length ? chips.join('') : '<span class="text-muted">none</span>';
}

/**
 * Rows of a hottest-positions table.
 * @param {object[]} positions - HotPosition list
 * @returns {string} HTML
 */
function hottestRows(positions) {
    const rows = (positions || []).slice(0, HOTTEST_SHOWN).map((item) => `<tr>
        <td class="text-mono">${escapeHtml(item.prompt + 1)}</td>
        <td class="text-mono">${escapeHtml(item.position)}</td>
        <td class="score-badge">${escapeHtml(formatNumber(item.kl))}</td>
        <td><span class="token-chip">${escapeHtml(visibleToken(item.context))}</span></td>
        <td>${tokenChip(item.base_token, 'down')} &rarr; ${tokenChip(item.tuned_token, 'up')}</td>
        <td>${copyButton(item.context)}</td></tr>`);
    return rows.length ? rows.join('') : '<tr><td colspan="6" class="text-muted">No position compared.</td></tr>';
}

/**
 * Side-by-side generations.
 * @param {object[]} generations - GenerationPair list
 * @returns {string} HTML
 */
function generationBlocks(generations) {
    return (generations || []).map((item) => `<div style="margin-bottom: 16px;">
        <div class="section-label text-break">${escapeHtml(preview(item.text))}</div>
        <div class="side-by-side">
            <div><div class="field-description">Base</div><pre class="output-block">${escapeHtml(item.base)}</pre></div>
            <div><div class="field-description">Tuned</div><pre class="output-block">${escapeHtml(item.tuned)}</pre></div>
        </div></div>`).join('');
}

/* ---------- Prompts file (§9.7) ---------- */

/**
 * Accept a selected or dropped file, refusing an oversized or unsupported one.
 * @param {File} file
 */
function loadFile(file) {
    const name = file ? file.name.toLowerCase() : '';
    if (!file) {
        notify('No file selected', 'error');
    } else if (file.size > MAX_FILE_SIZE) {
        notify('File size exceeds the 2 MB limit', 'error');
    } else if (!ALLOWED_SUFFIXES.some((suffix) => name.endsWith(suffix))) {
        notify(`Unsupported file: use ${ALLOWED_SUFFIXES.join(', ')}`, 'error');
    } else {
        selectedFile = file;
        document.getElementById('file-name').textContent = `${file.name} (${formatSize(file.size)})`;
        document.getElementById('file-drop-content').style.display = 'none';
        document.getElementById('file-drop-selected').style.display = '';
    }
}

/**
 * Reset the drop zone to its initial state.
 */
function resetFileZone() {
    selectedFile = null;
    document.getElementById('file-input').value = '';
    document.getElementById('file-drop-content').style.display = '';
    document.getElementById('file-drop-selected').style.display = 'none';
}

/**
 * Wire up the drop zone events.
 */
function initFileDropZone() {
    const dropZone = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');

    fileInput.addEventListener('change', (e) => loadFile(e.target.files[0]));
    document.getElementById('file-browse-link').addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.click();
    });
    document.getElementById('file-change-link').addEventListener('click', (e) => {
        e.preventDefault();
        resetFileZone();
        fileInput.click();
    });
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });
    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        loadFile(e.dataTransfer.files[0]);
    });
}

/**
 * Handle the batch run button click.
 */
async function handleRun() {
    if (!selectedFile) {
        notify('Choose a prompts file first', 'error');
    } else if (await ensureAllDetected([pickers.base, pickers.tuned])) {
        const payload = {
            base: pickers.base.value(),
            tuned: pickers.tuned.value(),
            field: document.getElementById('field-input').value.trim(),
            limit: document.getElementById('limit-input').value,
            chat: document.getElementById('chat-toggle').checked,
            generate_tokens: document.getElementById('generate-input').value,
        };
        const result = await runJob(BATCH_URL, POLL_URL, payload, 'Probing both models', selectedFile, null,
            (collapsed) => collapsePanels(INPUT_PANELS, collapsed));
        if (result) renderToolResult(result);
    }
}

/**
 * Reattach to a running or just-finished batch probe after a navigation.
 * Only the batch job is trackable, so the interactive session jobs never reach here; the
 * interactive state is restored separately by refreshSession.
 * @param {object} job - the current-job payload
 */
async function reattach(job) {
    collapsePanels(INPUT_PANELS, true);
    const result = await followJob(POLL_URL, job.job_id, job.label);
    if (result) {
        renderToolResult(result);
    } else {
        collapsePanels(INPUT_PANELS, false);
    }
}

/* ---------- Interactive session ---------- */

/**
 * The options of a session prompt or save.
 * @returns {object} {chat, top, generate_tokens}
 */
function sessionSettings() {
    return {
        chat: document.getElementById('session-chat').checked,
        top: document.getElementById('session-top').value,
        generate_tokens: document.getElementById('session-generate').value,
    };
}

/**
 * One history entry: the prompt, its KL, hottest positions, next tokens, shifts and generations.
 * @param {object} entry - {index, text, comparison, generation}
 * @returns {string} HTML
 */
function historyEntry(entry) {
    const comparison = entry.comparison;
    const generation = entry.generation ? generationBlocks([entry.generation]) : '';
    const nextBase = (comparison.next_base || []).map((item) =>
        tokenChip(item.text, '', `p ${formatNumber(item.probability)}`)).join('');
    const nextTuned = (comparison.next_tuned || []).map((item) =>
        tokenChip(item.text, '', `p ${formatNumber(item.probability)}`)).join('');
    return `<div class="history-entry">
        <h4>#${escapeHtml(entry.index + 1)} · mean KL <span class="score-badge">${escapeHtml(formatNumber(comparison.mean_kl))}</span>
            · max KL ${escapeHtml(formatNumber(comparison.max_kl))} · ${escapeHtml(comparison.positions)} positions</h4>
        <pre class="output-block">${escapeHtml(entry.text)}</pre>
        <div class="side-by-side">
            <div><div class="field-description">Next token, base</div><div class="token-list">${nextBase}</div></div>
            <div><div class="field-description">Next token, tuned</div><div class="token-list">${nextTuned}</div></div>
        </div>
        <div class="table-scroll" style="margin-top: 12px;">
            <table class="data-table" style="margin-bottom: 12px;">
                <thead><tr><th>Prompt</th><th>Position</th><th>KL</th><th>Context</th><th>Base &rarr; tuned</th><th style="width: 50px;"></th></tr></thead>
                <tbody>${hottestRows(comparison.hottest)}</tbody>
            </table>
        </div>
        <div class="field-description">Promoted</div><div class="token-list">${shiftChips(comparison.promoted, 'up')}</div>
        <div class="field-description" style="margin-top: 8px;">Suppressed</div><div class="token-list">${shiftChips(comparison.suppressed, 'down')}</div>
        ${generation}
    </div>`;
}

/**
 * Render the session state: loaded or not, placement, history.
 * @param {object} session - the session payload
 */
function renderSession(session) {
    const badge = document.getElementById('session-badge');
    badge.className = `badge ${session.loaded ? 'badge-success' : 'badge-neutral'}`;
    badge.textContent = session.loaded ? 'models loaded' : 'no models loaded';
    document.getElementById('unload-btn').style.display = session.loaded ? '' : 'none';
    document.getElementById('session-controls').style.display = session.loaded ? '' : 'none';
    if (session.loaded) {
        const placement = session.placement || {};
        const inputs = Object.entries(session.inputs).map(([key, value]) => `${key}: ${value}`).join(' · ');
        document.getElementById('session-info').textContent =
            `${inputs} — tuned on ${placement.tuned_device}, base on ${placement.base_device || 'the adapter host'} ` +
            `(${session.dtype}; ${placement.reason}). ${session.prompts} prompts analysed.`;
        const list = document.getElementById('history-list');
        list.innerHTML = session.history.slice().reverse().map(historyEntry).join('');
        wireCopyButtons(list);
    }
}

/**
 * Load the session as the server holds it (after a page reload, for instance).
 */
async function refreshSession() {
    const response = await apiFetch(SESSION_URL);
    if (response.status) {
        renderSession(response.data);
        if (response.data.loaded) {
            document.getElementById('mode-select').value = 'interactive';
            handleModeChange();
        }
    }
}

/**
 * Handle the load button click: load both models for the session.
 */
async function handleLoad() {
    if (await ensureAllDetected([pickers.base, pickers.tuned])) {
        const payload = { base: pickers.base.value(), tuned: pickers.tuned.value() };
        const result = await runJob(SESSION_URL, POLL_URL, payload, 'Loading both models');
        if (result) {
            renderSession(result.session);
            showSection('none');
            notify('Both models are loaded', 'success');
        }
    }
}

/**
 * Handle the compare button (or Ctrl+Enter): compare both models on the prompt.
 */
async function handlePrompt() {
    const text = document.getElementById('prompt-input').value;
    if (!text.trim()) {
        notify('Type a prompt first', 'error');
    } else {
        showSection('loading');
        setProgress('Comparing both models on the prompt', 'Running both models…', null);
        setRunning(true);
        document.getElementById('cancel-btn').style.display = 'none';
        const response = await apiFetch(PROMPT_URL, { method: 'POST', body: { text, ...sessionSettings() } });
        setRunning(false);
        if (response.status) {
            showSection('none');
            await refreshSession();
        } else {
            showError(response.message || 'The prompt could not be compared.');
        }
    }
}

/**
 * Handle the save button click: probe every session prompt and save the report.
 */
async function handleSave() {
    const result = await runJob(SAVE_URL, POLL_URL, sessionSettings(), 'Saving the session report');
    if (result) {
        const bar = document.getElementById('session-report-bar');
        bar.style.display = '';
        renderReportBar('session-report-bar', result.report, result.report_error);
        renderToolResult(result);
    }
}

/**
 * Handle the unload button click: release both models.
 */
async function handleUnload() {
    const response = await apiFetch(SESSION_URL, { method: 'DELETE' });
    if (response.status) {
        renderSession(response.data);
        document.getElementById('session-report-bar').style.display = 'none';
        notify('Models unloaded', 'info');
    } else {
        showError(response.message || 'The models could not be unloaded.');
    }
}

/**
 * Show the panel of the chosen mode.
 */
function handleModeChange() {
    const interactive = document.getElementById('mode-select').value === 'interactive';
    document.getElementById('batch-panel').style.display = interactive ? 'none' : '';
    document.getElementById('session-panel').style.display = interactive ? '' : 'none';
}

document.addEventListener('DOMContentLoaded', () => {
    pickers.base = new ModelPicker('base');
    pickers.tuned = new ModelPicker('tuned');
    initFileDropZone();
    document.getElementById('mode-select').addEventListener('change', handleModeChange);
    document.getElementById('run-btn').addEventListener('click', handleRun);
    document.getElementById('load-btn').addEventListener('click', handleLoad);
    document.getElementById('unload-btn').addEventListener('click', handleUnload);
    document.getElementById('prompt-btn').addEventListener('click', handlePrompt);
    document.getElementById('save-btn').addEventListener('click', handleSave);
    document.getElementById('prompt-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) handlePrompt();
    });
    refreshSession();
    initJobReattach('behaviour', reattach);
});
