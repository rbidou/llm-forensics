/* LLM Forensics - Analyse a model (the suspicion assessment workflow) */

/** The job endpoint of the workflow (POST to start, GET <id> to poll). */
const JOB_URL = '/api/forensics/assessment';

/** The model pickers of the page, created once the DOM is ready. */
const pickers = {};

/* Risk and severity mappings, mirroring reports/html.py so the live summary matches the report. */

/** Risk level -> .verdict modifier class. */
const RISK_CLASS = { clean: 'ok', low: 'info', elevated: 'warn', high: 'bad' };

/** Risk level -> caps heading of the verdict banner. */
const RISK_HEADING = { clean: 'NO RISK', low: 'LOW RISK', elevated: 'ELEVATED RISK', high: 'HIGH RISK' };

/** Risk level -> badge class of the executive summary's risk tag. */
const RISK_BADGE = { clean: 'badge-success', low: 'badge-info', elevated: 'badge-warning', high: 'badge-danger' };

/** Verdict confidence -> badge class. */
const CONFIDENCE_BADGE = { high: 'badge-success', normal: 'badge-info', low: 'badge-danger' };

/** Signal severity -> badge class. */
const SEVERITY_BADGE = { none: 'badge-neutral', low: 'badge-info', medium: 'badge-warning', high: 'badge-danger' };

/**
 * The .verdict modifier class of a risk level.
 * @param {string} level
 * @returns {string}
 */
function riskClass(level) {
    return RISK_CLASS[level] || 'none';
}

/**
 * The caps heading of a risk level ('NO RISK' for clean, '<LEVEL> RISK' otherwise).
 * @param {string} level
 * @returns {string}
 */
function riskHeading(level) {
    return RISK_HEADING[level] || `${String(level).toUpperCase()} RISK`;
}

/** The weight-analysis toggles, keyed by the recommendation field of the detection payload. */
const TOOL_TOGGLES = { weights: 'opt-weights', entropy: 'opt-entropy', shifts: 'opt-shifts', vocab: 'opt-vocab' };

/**
 * The live workflow tracker, in execution order. ``stages`` are the progress stages
 * (progress.py) that mark a step active; ``toggle`` is the checkbox that can skip it
 * (null for the always-run Source and Report steps).
 */
const WORKFLOW_STEPS = [
    { key: 'source', label: 'Source', stages: ['provenance', 'lineage'], toggle: null },
    { key: 'weights', label: 'Weight Diff', stages: ['weights'], toggle: 'opt-weights' },
    { key: 'entropy', label: 'Entropy', stages: ['entropy'], toggle: 'opt-entropy' },
    { key: 'shifts', label: 'Token Shifts', stages: ['shifts'], toggle: 'opt-shifts' },
    { key: 'vocab', label: 'Vocabulary', stages: ['vocab'], toggle: 'opt-vocab' },
    { key: 'behaviour', label: 'Behaviour', stages: ['behaviour'], toggle: 'opt-behaviour' },
    { key: 'report', label: 'Report', stages: ['report'], toggle: null },
];

/** Stages seen across every poll of the current run (events can arrive between polls). */
let seenStages = new Set();

/** The last stage that mapped to a step, kept while preamble stages ('assessment') show. */
let lastActiveIndex = -1;

/** Whether each WORKFLOW_STEPS entry was skipped, snapshotted when the run starts. */
let workflowSkipped = [];

/**
 * Show or hide the behaviour prompts field with its toggle.
 */
function syncBehaviour() {
    const on = document.getElementById('opt-behaviour').checked;
    document.getElementById('behaviour-prompts-group').style.display = on ? '' : 'none';
}

/**
 * Pre-select the analyses relevant for the detected model's format.
 * @param {object} data - the detection payload ({recommended: {weights, entropy, shifts, vocab}, ...})
 */
function applyRecommended(data) {
    const recommended = data && data.recommended;
    if (!recommended) return;
    Object.keys(TOOL_TOGGLES).forEach((tool) => {
        if (tool in recommended) document.getElementById(TOOL_TOGGLES[tool]).checked = recommended[tool];
    });
}

/**
 * The request body built from the form.
 * @returns {object}
 */
function buildPayload() {
    const text = (id) => document.getElementById(id).value;
    return {
        suspect: pickers.suspect.value(),
        source: pickers.source.value(),
        verify_source: document.getElementById('opt-verify').checked,
        search: document.getElementById('opt-search').checked,
        run_weights: document.getElementById('opt-weights').checked,
        run_entropy: document.getElementById('opt-entropy').checked,
        run_shifts: document.getElementById('opt-shifts').checked,
        run_vocab: document.getElementById('opt-vocab').checked,
        run_behaviour: document.getElementById('opt-behaviour').checked,
        prompts: text('opt-prompts'),
        generate_tokens: text('opt-generate'),
        candidates: text('opt-candidates'),
        entropy_layers: text('opt-entropy-layers'),
        entropy_bins: text('opt-entropy-bins'),
        updates: text('opt-updates'),
        tokens: text('opt-tokens'),
        layers: text('opt-layers'),
        top_k: text('opt-top-k'),
    };
}

/**
 * Render a finished assessment: the executive summary only, laid out as in the report,
 * with the Full Report and New Analysis buttons. The input panels stay collapsed.
 * @param {object} result - {analysis, report, report_error}
 */
function renderResults(result) {
    const data = result.analysis;
    const verdict = data.verdict || {};
    const executive = data.executive || {};
    const flagged = (data.signals || []).filter((signal) => signal.severity !== 'none');

    const banner = document.getElementById('verdict-banner');
    banner.className = `verdict ${riskClass(verdict.level)}`;
    banner.innerHTML = `<span class="verdict-risk">${escapeHtml(riskHeading(verdict.level))}</span>` +
        `<span class="verdict-detail">${escapeHtml(verdict.headline)}</span>`;

    document.getElementById('exec-body').innerHTML = execRows(verdict, executive, flagged);
    renderResultActions('result-actions', result, () => window.location.reload());
    showSection('results');
}

/**
 * The four rows of the executive table (mirrors the report's exec-table).
 * @param {object} verdict - {level, confidence, headline}
 * @param {object} executive - {source, impact, example}
 * @param {object[]} flagged - signals with a non-'none' severity
 * @returns {string} HTML
 */
function execRows(verdict, executive, flagged) {
    const riskClassName = RISK_BADGE[verdict.level] || 'badge-neutral';
    const confClassName = CONFIDENCE_BADGE[verdict.confidence] || 'badge-neutral';
    return `<tr><th>Risk &amp; confidence</th>` +
        `<td>${badgeHtml(riskClassName, riskHeading(verdict.level))} ` +
        `${badgeHtml(confClassName, 'confidence ' + verdict.confidence)}</td></tr>
        <tr><th>Source</th><td>${escapeHtml(executive.source || MISSING_VALUE)}</td></tr>
        <tr><th>Markers</th><td>${markersCell(flagged)}</td></tr>
        <tr><th>Impact &amp; example</th><td>${impactCell(executive)}</td></tr>`;
}

/**
 * A badge span with an escaped modifier class and label.
 * @param {string} cls - the badge modifier class
 * @param {string} text - the label
 * @returns {string} HTML
 */
function badgeHtml(cls, text) {
    return `<span class="badge ${escapeAttribute(cls)}">${escapeHtml(text)}</span>`;
}

/**
 * The Markers cell: one line per flagged signal, or a reassurance when none were raised.
 * @param {object[]} flagged - signals with a non-'none' severity
 * @returns {string} HTML
 */
function markersCell(flagged) {
    let cell = '<span class="text-muted">Every analysis stayed within its baseline.</span>';
    if (flagged.length) {
        const chips = flagged.map((signal) =>
            `<li>${badgeHtml(SEVERITY_BADGE[signal.severity] || 'badge-neutral', signal.severity)} ` +
            `${escapeHtml(signal.title)}</li>`).join('');
        cell = `<ul class="marker-list">${chips}</ul>`;
    }
    return cell;
}

/**
 * The Impact & example cell: the plain-language impact and example, or a note when there is none.
 * @param {object} executive - {impact, example}
 * @returns {string} HTML
 */
function impactCell(executive) {
    const impact = executive.impact ? `<p><strong>Impact:</strong> ${escapeHtml(executive.impact)}</p>` : '';
    const example = executive.example ? `<p><strong>Example:</strong> ${escapeHtml(executive.example)}</p>` : '';
    return impact || example ? `${impact}${example}` : '<span class="text-muted">No material impact for the flags raised.</span>';
}

/** The input panels folded away while a run is in progress, in top-to-bottom order. */
const COLLAPSIBLE_PANELS = ['model-panel', 'workflow-panel'];

/**
 * Collapse or expand both input panels (Model and Workflow) together.
 * @param {boolean} collapsed
 */
function collapseInputs(collapsed) {
    collapsePanels(COLLAPSIBLE_PANELS, collapsed);
}

/**
 * Snapshot which steps are skipped and draw the tracker with every step pending.
 */
function resetSteps() {
    seenStages = new Set();
    lastActiveIndex = -1;
    workflowSkipped = WORKFLOW_STEPS.map((step) => Boolean(step.toggle) && !document.getElementById(step.toggle).checked);
    renderSteps(WORKFLOW_STEPS.map((step, i) => (workflowSkipped[i] ? 'skipped' : 'pending')));
}

/**
 * Update the tracker from one poll of the running job.
 * @param {object} data - the job payload ({progress:{stage}, events:[{stage}], completed, state})
 */
function updateSteps(data) {
    renderSteps(computeStatuses(data));
}

/**
 * The status of every step given the stages seen so far.
 * A step is running when the latest stage is one of its own; executed once its stage
 * (or a later step's) has been seen; pending until then; skipped when its toggle was off.
 * @param {object} data - the job payload
 * @returns {string[]} one of 'pending'|'running'|'executed'|'skipped' per step
 */
function computeStatuses(data) {
    (data.events || []).forEach((event) => seenStages.add(event.stage));
    const activeStage = data.progress ? data.progress.stage : null;
    const activeIndex = WORKFLOW_STEPS.findIndex((step) => step.stages.includes(activeStage));
    if (activeIndex >= 0) lastActiveIndex = activeIndex;
    const succeeded = data.completed && data.state === 'succeeded';
    const runningIndex = data.completed ? -1 : lastActiveIndex;
    return WORKFLOW_STEPS.map((step, i) => {
        let status = 'pending';
        if (workflowSkipped[i]) {
            status = 'skipped';
        } else if (succeeded || i < runningIndex || step.stages.some((stage) => seenStages.has(stage))) {
            status = 'executed';
        }
        if (!workflowSkipped[i] && !succeeded && i === runningIndex) status = 'running';
        return status;
    });
}

/**
 * The incoming-connector class for each step.
 * A non-skipped step's line is coloured by its own status (in CSS). A skipped step has no
 * status of its own, so its line borrows the next non-skipped step's: grey while that step
 * is still pending (the run has not reached it), orange while it runs and green once it has
 * finished. This draws one unbroken line from the last executed step, through the skipped
 * step(s), to the step that actually runs.
 * @param {string[]} statuses - 'pending'|'running'|'executed'|'skipped', one per step
 * @returns {string[]} '', ' is-link-running' or ' is-link-done' per step
 */
function skipConnectorClasses(statuses) {
    return statuses.map((status, i) => {
        let cls = '';
        if (status === 'skipped') {
            const next = statuses.slice(i + 1).find((other) => other !== 'skipped');
            if (next === 'running') {
                cls = ' is-link-running';
            } else if (next === 'executed') {
                cls = ' is-link-done';
            }
        }
        return cls;
    });
}

/**
 * Draw the step tracker from a status per step.
 * @param {string[]} statuses - 'pending'|'running'|'executed'|'skipped', one per WORKFLOW_STEPS entry
 */
function renderSteps(statuses) {
    const marks = { executed: '✓', skipped: '–' };
    const links = skipConnectorClasses(statuses);
    document.getElementById('workflow-steps').innerHTML = WORKFLOW_STEPS.map((step, i) => {
        const status = statuses[i];
        const mark = marks[status] || String(i + 1);
        const linkClass = links[i];
        return `<li class="workflow-step is-${escapeAttribute(status)}${linkClass}" data-step="${escapeAttribute(step.key)}">
            <span class="step-node" aria-hidden="true">${escapeHtml(mark)}</span>
            <span class="step-text">
                <span class="step-label">${escapeHtml(step.label)}</span>
                <span class="step-state">${escapeHtml(status)}</span>
            </span>
        </li>`;
    }).join('');
}

/**
 * Handle the run button click: detect the models, then run the workflow.
 */
async function handleRun() {
    const needed = [pickers.suspect];
    if (pickers.source.value().ref) needed.push(pickers.source);
    if (await ensureAllDetected(needed)) {
        const result = await runJob(JOB_URL, JOB_URL, buildPayload(), 'Analysing the model', null, updateSteps,
            (collapsed) => { if (collapsed) resetSteps(); collapseInputs(collapsed); });
        if (result) renderResults(result);   // success: keep the inputs collapsed behind the summary
    }
}

/**
 * Reattach to a running or just-finished assessment after a navigation.
 * The step tracker starts with nothing marked skipped (the launch form's skips are not known
 * on a return); it fills from the job's events and is replaced by the summary once it finishes.
 * @param {object} job - the current-job payload
 */
async function reattach(job) {
    seenStages = new Set();
    lastActiveIndex = -1;
    workflowSkipped = WORKFLOW_STEPS.map(() => false);
    renderSteps(WORKFLOW_STEPS.map(() => 'pending'));
    collapseInputs(true);
    const result = await followJob(JOB_URL, job.job_id, job.label, updateSteps);
    if (result) {
        renderResults(result);
    } else {
        collapseInputs(false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    pickers.suspect = new ModelPicker('suspect', applyRecommended);
    pickers.source = new ModelPicker('source');
    document.getElementById('opt-behaviour').addEventListener('change', syncBehaviour);
    document.getElementById('run-btn').addEventListener('click', handleRun);
    initJobReattach('assessment', reattach);
});
