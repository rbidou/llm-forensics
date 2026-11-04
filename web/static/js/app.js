/* LLM Forensics - Shared JavaScript utilities */

/** Milliseconds between two polls of a running job (§9.13). */
const POLL_INTERVAL_MS = 1000;

/** Milliseconds between two polls of the shared job watcher (cross-page finish notifications). */
const WATCH_INTERVAL_MS = 2000;

/** The endpoint describing the running (or most recent) job, for reattach and notifications. */
const CURRENT_JOB_URL = '/api/forensics/jobs/current';

/** Milliseconds between two polls of the live menu-bar system readout (RAM and VRAM stay current). */
const SYSTEM_INFO_INTERVAL_MS = 5000;

/** The endpoint describing the listening address, RAM and GPU, for the menu-bar readout. */
const SYSTEM_INFO_URL = '/api/forensics/system';

/** Menu-bar status condition -> [label text, status-dot modifier class]. */
const SYSTEM_STATUS = {
    running: ['Running', 'running'],
    error: ['Error', 'offline'],
    ok: ['OK', 'online'],
};

/** Shorter status-bar names for running jobs whose full label is too long for the menu bar. */
const RUNNING_LABELS = {
    'Vocabulary & Embeddings': 'Vocabulary',
};

/** Status-bar name by job kind, for a running job that carries no label (the interactive Behaviour Probe). */
const RUNNING_KIND_LABELS = {
    behaviour: 'Behaviour Probe',
};

/** Shown for VRAM and the GPU when the machine has no CUDA device. */
const NOT_APPLICABLE = 'N/A';

/** sessionStorage key: the id of the job whose finish has already been announced (shown once). */
const NOTIFIED_JOB_KEY = 'llmf.notifiedJob';

/** sessionStorage key: the id of the job whose result has already been shown on its own page. */
const SHOWN_RESULT_KEY = 'llmf.shownResultJob';

/** Terminal job state -> the word used in its finish notification. */
const FINISH_WORD = { succeeded: 'finished', failed: 'failed', cancelled: 'cancelled' };

/** Terminal job state -> the notification tone. */
const FINISH_TONE = { succeeded: 'success', failed: 'error', cancelled: 'info' };

/** How whitespace is shown inside a token chip. */
const VISIBLE_WHITESPACE = { ' ': '␣', '\n': '\\n', '\t': '\\t', '\r': '\\r' };

/** Shown for an empty token. */
const EMPTY_TOKEN = '∅';

/** Shown for a missing value. */
const MISSING_VALUE = '—';

/** The id of the job this page is running, for the Cancel button. */
let activeJobId = null;

/**
 * Fetch JSON from an API endpoint.
 * Object bodies are serialized automatically.
 * @param {string} url
 * @param {object} options - fetch options
 * @returns {Promise<object>} the parsed API envelope
 */
async function apiFetch(url, options = {}) {
    const defaults = {
        headers: { 'Content-Type': 'application/json' },
    };
    const merged = { ...defaults, ...options };
    if (merged.body && typeof merged.body === 'object') {
        merged.body = JSON.stringify(merged.body);
    }
    const response = await fetch(url, merged);
    return response.json();
}

/**
 * Post a multipart form (the browser sets the Content-Type and its boundary).
 * @param {string} url
 * @param {FormData} formData
 * @returns {Promise<object>} the parsed API envelope
 */
async function apiFetchForm(url, formData) {
    const response = await fetch(url, { method: 'POST', body: formData });
    return response.json();
}

/**
 * Show a transient notification toast in the top-right corner.
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 */
function notify(message, type = 'info') {
    const el = document.createElement('div');
    el.className = `notification ${type}`;
    el.textContent = message;
    document.body.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity 0.3s';
        setTimeout(() => el.remove(), 300);
    }, 3000);
}

/**
 * Show a notification toast carrying a trailing link (built from DOM nodes, never innerHTML).
 * @param {string} message
 * @param {'success'|'error'|'info'} type
 * @param {string} linkText - the link's visible text
 * @param {?string} href - the link target; the link is omitted when null
 */
function notifyWithLink(message, type, linkText, href) {
    const el = document.createElement('div');
    el.className = `notification ${type}`;
    el.appendChild(document.createTextNode(`${message} `));
    if (href) {
        const link = document.createElement('a');
        link.textContent = linkText;
        link.setAttribute('href', href);
        el.appendChild(link);
    }
    document.body.appendChild(el);
    setTimeout(() => {
        el.style.opacity = '0';
        el.style.transition = 'opacity 0.3s';
        setTimeout(() => el.remove(), 300);
    }, 5000);
}

/**
 * Copy text to the clipboard and confirm with a notification.
 * @param {string} text
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        notify('Copied to clipboard', 'success');
    }).catch(() => {
        notify('Failed to copy', 'error');
    });
}

/**
 * Escape HTML special characters for safe interpolation into innerHTML.
 * @param {*} str - converted to text; null and undefined become empty
 * @returns {string}
 */
function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str === null || str === undefined ? '' : String(str);
    return div.innerHTML;
}

/**
 * Escape a value for a double-quoted HTML attribute: escapeHtml() leaves quotes,
 * which delimit the attribute, untouched.
 * @param {*} str
 * @returns {string}
 */
function escapeAttribute(str) {
    return escapeHtml(str).replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

/**
 * Trigger a client-side file download.
 * @param {string|Blob} content
 * @param {string} filename
 * @param {string} mimeType
 */
function downloadFile(content, filename, mimeType = 'application/octet-stream') {
    const blob = content instanceof Blob ? content : new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
}

/**
 * Download an object as a pretty-printed JSON file.
 * @param {*} data
 * @param {string} filename
 */
function downloadJson(data, filename) {
    downloadFile(JSON.stringify(data, null, 2), filename, 'application/json');
}

/**
 * Format a byte count in human-readable form.
 * @param {number} bytes
 * @returns {string}
 */
function formatSize(bytes) {
    let text = `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    if (bytes < 1024) {
        text = `${bytes} B`;
    } else if (bytes < 1024 * 1024) {
        text = `${(bytes / 1024).toFixed(1)} KB`;
    }
    return text;
}

/**
 * Debounce a function call.
 * @param {Function} fn
 * @param {number} delay - milliseconds
 * @returns {Function}
 */
function debounce(fn, delay = 300) {
    let timer = null;
    return function (...args) {
        clearTimeout(timer);
        timer = setTimeout(() => fn.apply(this, args), delay);
    };
}

/* ---------- Value formatting ---------- */

/**
 * Format a number with a fixed number of significant digits ('—' when missing).
 * @param {number|null|undefined} value
 * @param {number} digits - significant digits
 * @returns {string}
 */
function formatNumber(value, digits = 4) {
    let text = MISSING_VALUE;
    if (typeof value === 'number' && Number.isFinite(value)) {
        text = Number.isInteger(value) ? String(value) : Number(value.toPrecision(digits)).toString();
    }
    return text;
}

/**
 * Format a fraction as a percentage ('—' when missing).
 * @param {number|null|undefined} value - between 0 and 1
 * @returns {string}
 */
function formatPercent(value) {
    return typeof value === 'number' ? `${(value * 100).toFixed(1)}%` : MISSING_VALUE;
}

/**
 * Make a token's whitespace visible ('␣' for a space, '\n' for a newline).
 * @param {string} text
 * @returns {string}
 */
function visibleToken(text) {
    const value = text === null || text === undefined ? '' : String(text);
    return value === '' ? EMPTY_TOKEN : Array.from(value).map((ch) => VISIBLE_WHITESPACE[ch] || ch).join('');
}

/**
 * A token chip, escaped, with its whitespace visible.
 * @param {string} text - the raw token text
 * @param {string} extraClass - '', 'up' or 'down'
 * @param {string} title - tooltip (a score, for instance)
 * @returns {string} HTML
 */
function tokenChip(text, extraClass = '', title = '') {
    return `<span class="token-chip ${escapeAttribute(extraClass)}" title="${escapeAttribute(title)}">` +
        `${escapeHtml(visibleToken(text))}</span>`;
}

/**
 * A badge for a z-score against a null baseline: z ≥ 3 stands out, z ≥ 1 is weak.
 * @param {number|null|undefined} z
 * @returns {string} HTML
 */
function zBadge(z) {
    let badgeClass = 'badge-neutral';
    if (typeof z === 'number' && z >= 3) {
        badgeClass = 'badge-success';
    } else if (typeof z === 'number' && z >= 1) {
        badgeClass = 'badge-warning';
    }
    return `<span class="badge ${badgeClass}">z ${escapeHtml(formatNumber(z, 3))}</span>`;
}

/**
 * A trailing copy button for a table row.
 * @param {string} value - what is copied
 * @returns {string} HTML
 */
function copyButton(value) {
    return `<button class="btn btn-sm copy-btn" data-value="${escapeAttribute(value)}" title="Copy">&#9112;</button>`;
}

/**
 * Wire every copy button inside a container.
 * @param {HTMLElement} container
 */
function wireCopyButtons(container) {
    container.querySelectorAll('.copy-btn').forEach((btn) => {
        btn.addEventListener('click', () => copyToClipboard(btn.dataset.value));
    });
}

/**
 * Render a list of messages (warnings, notes) into a <ul>, hiding its block when empty.
 * @param {string} listId - the <ul>
 * @param {string} blockId - the element shown only when there are messages
 * @param {string[]} messages
 */
function renderMessages(listId, blockId, messages) {
    const items = messages || [];
    document.getElementById(listId).innerHTML = items.map((text) => `<li>${escapeHtml(text)}</li>`).join('');
    document.getElementById(blockId).style.display = items.length ? '' : 'none';
}

/* ---------- The standard page regions ---------- */

/**
 * Show one page section and hide the others.
 * @param {'loading'|'results'|'error'|'none'} section
 */
function showSection(section) {
    ['loading', 'results', 'error'].forEach((name) => {
        const el = document.getElementById(`${name}-section`);
        if (el) el.style.display = section === name ? '' : 'none';
    });
}

/**
 * Render an API failure inline, where the page recipe puts it.
 * @param {string} message
 */
function showError(message) {
    document.getElementById('error-message').textContent = message;
    showSection('error');
}

/* ---------- Saved reports ---------- */

/**
 * Show the report saved by a finished analysis: its id, a view link and the downloads.
 * @param {string} containerId
 * @param {object|null} saved - {id, view_url, html_url, json_url}
 * @param {string} saveError - why the report could not be saved, if it was not
 */
function renderReportBar(containerId, saved, saveError) {
    const container = document.getElementById(containerId);
    let html = `<span class="text-danger">${escapeHtml(saveError || 'No report was saved.')}</span>`;
    if (saved) {
        html = `<span class="report-id" title="Saved report">${escapeHtml(saved.id)}</span>
            <a class="btn btn-sm" href="${escapeAttribute(saved.view_url)}" target="_blank" rel="noopener noreferrer">Open report</a>
            <a class="btn btn-sm" href="${escapeAttribute(saved.html_url)}" download>Download HTML</a>
            <a class="btn btn-sm" href="${escapeAttribute(saved.json_url)}" download>Download JSON</a>`;
    }
    container.innerHTML = html;
}

/* ---------- The finished tool view (collapsed inputs, summary, actions) ----------
   A finished tool run folds its input panel(s) away and shows only a verdict banner
   (the analysis summary) and two buttons. The report keeps every detail. These helpers
   are shared by every tool page and the home workflow. */

/**
 * Collapse or expand one panel, keeping its header toggle in sync.
 * @param {string} panelId
 * @param {boolean} collapsed
 */
function setPanelCollapsed(panelId, collapsed) {
    const panel = document.getElementById(panelId);
    if (!panel) return;
    panel.classList.toggle('collapsed', collapsed);
    const toggle = panel.querySelector('.panel-toggle');
    if (toggle) {
        toggle.setAttribute('aria-expanded', String(!collapsed));
        toggle.setAttribute('aria-label', `${collapsed ? 'Expand' : 'Collapse'} ${toggle.dataset.panelName}`);
    }
}

/**
 * Collapse or expand several input panels together.
 * @param {string[]} panelIds - top-to-bottom order
 * @param {boolean} collapsed
 */
function collapsePanels(panelIds, collapsed) {
    panelIds.forEach((panelId) => setPanelCollapsed(panelId, collapsed));
}

/**
 * Wire every collapsible panel's header toggle: a click folds or unfolds its panel.
 */
function initPanelToggles() {
    document.querySelectorAll('.panel-toggle').forEach((toggle) => {
        const panel = toggle.closest('.panel');
        toggle.addEventListener('click', () => setPanelCollapsed(panel.id, !panel.classList.contains('collapsed')));
    });
}

/**
 * Render an analysis summary into a verdict banner (coloured by its tone).
 * @param {string} bannerId
 * @param {object} summary - {tone, heading, headline}
 */
function renderVerdict(bannerId, summary) {
    const data = summary || {};
    const banner = document.getElementById(bannerId);
    banner.className = `verdict ${data.tone || 'none'}`;
    banner.innerHTML = `<span class="verdict-risk">${escapeHtml(data.heading)}</span>` +
        `<span class="verdict-detail">${escapeHtml(data.headline)}</span>`;
}

/**
 * Render a summary's supporting lines into a list, hiding it when there are none.
 * @param {string} listId
 * @param {string[]} lines
 */
function renderSummaryLines(listId, lines) {
    const items = lines || [];
    const list = document.getElementById(listId);
    if (!list) return;
    list.innerHTML = items.map((line) => `<li>${escapeHtml(line)}</li>`).join('');
    list.style.display = items.length ? '' : 'none';
}

/**
 * Draw the finished-run action buttons: Open report (the saved report page) and New analysis.
 * @param {string} containerId
 * @param {object} result - {report, report_error}
 * @param {function(): void} onNewAnalysis - run when New analysis is clicked
 */
function renderResultActions(containerId, result, onNewAnalysis) {
    const saved = result.report;
    const openReport = saved
        ? `<a class="btn btn-primary" href="${escapeAttribute(saved.view_url)}" target="_blank" rel="noopener noreferrer">Open report</a>`
        : `<span class="text-danger">${escapeHtml(result.report_error || 'No report was saved.')}</span>`;
    const container = document.getElementById(containerId);
    container.innerHTML = `${openReport}<button class="btn" type="button" id="new-analysis-btn">New analysis</button>`;
    document.getElementById('new-analysis-btn').addEventListener('click', onNewAnalysis);
}

/**
 * Render a finished tool run: the summary banner and lines, the action buttons, and show it.
 * @param {object} result - {analysis:{summary}, report, report_error}
 */
function renderToolResult(result) {
    const summary = result.analysis.summary;
    renderVerdict('results-verdict', summary);
    renderSummaryLines('summary-lines', summary.lines);
    renderResultActions('result-actions', result, () => window.location.reload());
    showSection('results');
}

/* ---------- Long-running jobs (§9.13) ----------
   The client chooses the job id, posts the job, then polls it every second,
   rendering the latest progress until the payload says it is completed. */

/**
 * Update the progress panel.
 * @param {string} title
 * @param {string} message
 * @param {number|null} fraction - null when the analysis cannot report a total
 */
function setProgress(title, message, fraction) {
    const known = fraction !== null && fraction !== undefined;
    document.getElementById('progress-title').textContent = title;
    document.getElementById('progress-message').textContent = message;
    const fill = document.getElementById('progress-fill');
    fill.classList.toggle('indeterminate', !known);
    fill.style.width = known ? `${fraction * 100}%` : '';
}

/**
 * Enable or disable every run button of the page, and show the Cancel button while running.
 * @param {boolean} running
 */
function setRunning(running) {
    document.querySelectorAll('[data-run]').forEach((btn) => {
        btn.disabled = running || btn.dataset.disabledFeature === 'true';
    });
    const cancel = document.getElementById('cancel-btn');
    if (cancel) cancel.style.display = running ? '' : 'none';
}

/**
 * Poll a job until it reports completion.
 * @param {string} url - the job's polling URL
 * @param {string} title - shown in the progress panel
 * @param {?function(object): void} onProgress - called with each poll's job payload (the final one too)
 * @returns {Promise<object|null>} the final job payload, or null when polling failed
 */
function pollJob(url, title, onProgress = null) {
    return new Promise((resolve) => {
        let cursor = 0;
        const pollTimer = setInterval(async () => {
            try {
                const response = await apiFetch(`${url}?cursor=${cursor}`);
                if (!response.status) {
                    clearInterval(pollTimer);
                    showError(response.message || 'Failed to fetch the job state.');
                    resolve(null);
                    return;
                }
                cursor = response.data.cursor;
                setProgress(title, response.data.progress.message, response.data.progress.fraction);
                if (onProgress) onProgress(response.data);
                if (response.data.completed) {
                    clearInterval(pollTimer);
                    resolve(response.data);
                }
            } catch (err) {
                clearInterval(pollTimer);
                showError(`Polling error: ${err.message}`);
                resolve(null);
            }
        }, POLL_INTERVAL_MS);
    });
}

/**
 * Announce a tracked process's finish once, wherever the user is.
 * Deduplicated across page loads through sessionStorage, so returning to a page does not
 * re-announce a job already seen. Untracked jobs (the interactive session) stay silent.
 * @param {?object} job - a job or current-job payload ({job_id, label, trackable, completed, state})
 */
function maybeNotifyFinish(job) {
    if (!job || !job.trackable || !job.completed) return;
    if (sessionStorage.getItem(NOTIFIED_JOB_KEY) === job.job_id) return;
    sessionStorage.setItem(NOTIFIED_JOB_KEY, job.job_id);
    notify(`${job.label} ${FINISH_WORD[job.state] || 'finished'}`, FINISH_TONE[job.state] || 'info');
}

/**
 * Follow a job already started (by this page or before a navigation) through to its outcome.
 * Failures are rendered inline; the caller only handles success.
 * @param {string} pollUrl - the polling prefix (the job id is appended)
 * @param {string} jobId - the job to follow
 * @param {string} title - shown in the progress panel
 * @param {?function(object): void} onProgress - forwarded to pollJob (each poll's job payload)
 * @returns {Promise<object|null>} the job's result, or null on any failure
 */
async function followJob(pollUrl, jobId, title, onProgress = null) {
    let result = null;
    showSection('loading');
    setProgress(title, 'Starting…', null);
    setRunning(true);
    activeJobId = jobId;
    const job = await pollJob(`${pollUrl}/${jobId}`, title, onProgress);
    activeJobId = null;
    if (job) {
        sessionStorage.setItem(SHOWN_RESULT_KEY, jobId);  // its outcome has now been shown on this page
        maybeNotifyFinish(job);
        if (job.state === 'succeeded') {
            result = job.result;
        } else {
            showError(job.cancelled ? 'Cancelled.' : (job.error || `${title} failed.`));
        }
    }
    setRunning(false);
    return result;
}

/**
 * Start a job and follow it through to its outcome.
 * The start is posted before anything on screen changes, so a refused start (another process
 * already running) leaves the inputs and sections exactly as they were, explained by a single
 * notification. Only once the start is accepted does onStart fold the inputs away and the
 * progress panel appear. Other failures are rendered inline.
 * @param {string} startUrl - the POST endpoint
 * @param {string} pollUrl - the polling prefix (the job id is appended)
 * @param {object} payload - the request body (job_id is added)
 * @param {string} title - shown in the progress panel
 * @param {File|null} file - sent as a multipart upload with the payload as 'params'
 * @param {?function(object): void} onProgress - forwarded to pollJob (each poll's job payload)
 * @param {?function(boolean): void} toggleInputs - fold the page's inputs away (true) once the start
 *     is accepted, and reopen them (false) if the run then fails or is cancelled; never called on a refusal
 * @returns {Promise<object|null>} the job's result, or null on any failure
 */
async function runJob(startUrl, pollUrl, payload, title, file = null, onProgress = null, toggleInputs = null) {
    const jobId = newJobId();
    const body = { ...payload, job_id: jobId };
    let result = null;

    let response = null;
    if (file) {
        const formData = new FormData();
        formData.append('file', file);
        formData.append('params', JSON.stringify(body));
        response = await apiFetchForm(startUrl, formData);
    } else {
        response = await apiFetch(startUrl, { method: 'POST', body });
    }

    if (!response.status) {
        await handleStartFailure(response, title);   // refused: leave the inputs and sections untouched
    } else {
        if (toggleInputs) toggleInputs(true);   // accepted: only now fold the inputs away and show progress
        result = await followJob(pollUrl, jobId, title, onProgress);
        if (!result && toggleInputs) toggleInputs(false);   // failed or cancelled: reopen the inputs
    }
    return result;
}

/**
 * Explain a refused start: a busy process gets a notification naming it (with a link to it),
 * anything else the plain inline error.
 * @param {object} response - the failed start envelope
 * @param {string} title - the process the user tried to start
 */
async function handleStartFailure(response, title) {
    const current = await apiFetch(CURRENT_JOB_URL);
    const job = current.status ? current.data.job : null;
    if (job && !job.completed) {
        const where = job.progress && job.progress.message ? ` (${job.progress.message})` : '';
        notifyWithLink(`${job.label || 'A process'} is already running${where}. Wait for it or cancel it.`,
            'error', 'Go to it', job.url);
    } else {
        showError(response.message || `${title} could not start.`);
    }
}

/**
 * A random 32-character hex job id. crypto.randomUUID only exists in secure contexts
 * (HTTPS or localhost), so a server reached over plain HTTP on a LAN address needs this.
 * @returns {string}
 */
function newJobId() {
    const bytes = crypto.getRandomValues(new Uint8Array(16));
    return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}

/**
 * Ask the running job to stop at its next checkpoint.
 */
async function handleCancel() {
    if (!activeJobId) return;
    const response = await apiFetch(`/api/forensics/jobs/${encodeURIComponent(activeJobId)}/cancel`, { method: 'POST' });
    notify(response.status ? 'Cancelling: waiting for the analysis to reach a checkpoint' : response.message, 'info');
}

/**
 * Open or close the drawer menu of the compact layout.
 * @param {boolean} open
 * @param {boolean} restoreFocus - give the focus back to the toggle when closing
 */
function setNavOpen(open, restoreFocus = true) {
    const toggle = document.getElementById('nav-toggle');
    const wasOpen = document.body.classList.contains('nav-open');
    document.body.classList.toggle('nav-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close the menu' : 'Open the menu');
    if (open) {
        const first = document.querySelector('#sidebar .sidebar-nav a');
        if (first) setTimeout(() => first.focus(), 50);
    } else if (wasOpen && restoreFocus) {
        toggle.focus();
    }
}

/**
 * Wire the drawer menu of the compact layout (< 1400px): toggle button, backdrop, Esc,
 * closing on navigation and when the window becomes wide again.
 */
function initNavDrawer() {
    const toggle = document.getElementById('nav-toggle');
    if (!toggle) return;
    toggle.addEventListener('click', () => setNavOpen(!document.body.classList.contains('nav-open')));
    document.getElementById('nav-backdrop').addEventListener('click', () => setNavOpen(false));
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && document.body.classList.contains('nav-open')) setNavOpen(false);
    });
    document.querySelectorAll('#sidebar a').forEach((link) => {
        link.addEventListener('click', () => setNavOpen(false, false));
    });
    // Same breakpoint as the compact layout in app.css
    window.matchMedia('(min-width: 1400px)').addEventListener('change', (e) => {
        if (e.matches) setNavOpen(false, false);
    });
}

/* ---------- Running process awareness (reattach + cross-page notifications) ---------- */

/**
 * Watch the shared job on every page: announce each tracked process's finish once,
 * wherever the user has navigated. The page that launched the job also announces it
 * through followJob; maybeNotifyFinish deduplicates so only one toast shows.
 * A job already finished before this page opened is adopted silently on the first poll,
 * so a freshly opened tab never announces a stale finish.
 */
function initJobWatcher() {
    let first = true;
    const tick = async () => {
        try {
            const response = await apiFetch(CURRENT_JOB_URL);
            if (!response.status) return;
            const job = response.data.job;
            if (first && job && job.trackable && job.completed) {
                sessionStorage.setItem(NOTIFIED_JOB_KEY, job.job_id);
            } else {
                maybeNotifyFinish(job);
            }
            updateSystemStatus(job);
        } catch (err) {
            // A transient poll failure is ignored: the next tick tries again.
        } finally {
            first = false;
        }
    };
    tick();
    setInterval(tick, WATCH_INTERVAL_MS);
}

/**
 * Reattach a returning tool page to its own running (or just-finished) process.
 * The page provides how to display it; a finished job already seen on this page is left
 * alone so the inputs show fresh.
 * @param {string} kind - this page's job kind
 * @param {function(object): void} render - given the current-job payload, resume the display
 */
async function initJobReattach(kind, render) {
    const response = await apiFetch(CURRENT_JOB_URL);
    if (!response.status) return;
    const job = response.data.job;
    if (!job || job.kind !== kind || !job.trackable) return;
    if (job.completed && sessionStorage.getItem(SHOWN_RESULT_KEY) === job.job_id) return;
    render(job);
}

/* ---------- Menu-bar system status and information ---------- */

/**
 * Set an element's text content by id, when the element is present.
 * @param {string} id
 * @param {string} text
 */
function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
}

/**
 * Format a byte count as gibibytes with one decimal (e.g. '15.8').
 * @param {number} bytes
 * @returns {string}
 */
function formatGb(bytes) {
    return (bytes / (1024 * 1024 * 1024)).toFixed(1);
}

/**
 * Format a mebibyte count as gibibytes with one decimal (nvidia-smi reports VRAM in MiB).
 * @param {number} mib
 * @returns {string}
 */
function formatMibAsGb(mib) {
    return (mib / 1024).toFixed(1);
}

/**
 * Set the menu-bar status chip from the shared job: Running while one is live, Error when the
 * most recent one failed, OK otherwise.
 * @param {?object} job - a current-job payload ({completed, state}) or null when none has run
 */
function updateSystemStatus(job) {
    const chip = document.getElementById('system-status-chip');
    const label = document.getElementById('system-status-label');
    if (!chip || !label) return;
    let key = 'ok';
    if (job && !job.completed) {
        key = 'running';
    } else if (job && job.state === 'failed') {
        key = 'error';
    }
    const [text, modifier] = SYSTEM_STATUS[key];
    chip.className = `status-dot ${modifier}`;
    // While running, name the task: a shortened label where defined, else the job's own label,
    // else a name for its kind (the interactive Behaviour Probe jobs carry no label).
    const name = key === 'running' ? RUNNING_LABELS[job.label] || job.label || RUNNING_KIND_LABELS[job.kind] : '';
    label.textContent = name ? `${text}: ${name}` : text;
}

/**
 * Render one system-information readout into the menu bar (text only, never innerHTML).
 * @param {object} data - {address, ram, gpu}
 */
function renderSystemInfo(data) {
    const address = data.address || {};
    setText('system-ip', address.host ? `${address.host}:${address.port}` : MISSING_VALUE);
    const ram = data.ram;
    setText('system-ram', ram ? `${formatGb(ram.available_bytes)} / ${formatGb(ram.total_bytes)} GB` : MISSING_VALUE);
    const gpu = data.gpu;
    setText('system-gpu', gpu ? gpu.name : NOT_APPLICABLE);
    setText('system-vram',
        gpu ? `${formatMibAsGb(gpu.free_vram_mb)} / ${formatMibAsGb(gpu.total_vram_mb)} GB` : NOT_APPLICABLE);
}

/**
 * Fetch the system-information readout now and on a fixed interval, keeping RAM and VRAM live.
 */
function initSystemInfo() {
    const refresh = async () => {
        try {
            const response = await apiFetch(SYSTEM_INFO_URL);
            if (response.status) renderSystemInfo(response.data);
        } catch (err) {
            // A transient failure is ignored: the next refresh tries again.
        }
    };
    refresh();
    setInterval(refresh, SYSTEM_INFO_INTERVAL_MS);
}

/**
 * Wire the collapsible sidebar sections: each header toggles its category open or shut.
 * The category holding the active page is already open (set server-side), so nothing flashes.
 */
function initSidebarMenu() {
    document.querySelectorAll('.nav-category-header').forEach((header) => {
        header.addEventListener('click', () => {
            const open = header.parentElement.classList.toggle('open');
            header.setAttribute('aria-expanded', String(open));
        });
    });
}

document.addEventListener('DOMContentLoaded', () => {
    const cancel = document.getElementById('cancel-btn');
    if (cancel) cancel.addEventListener('click', handleCancel);
    initNavDrawer();
    initSidebarMenu();
    initPanelToggles();
    initJobWatcher();
    initSystemInfo();
});
