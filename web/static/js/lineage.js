/* LLM Forensics - Hub Lineage
   Two jobs: the discovery reads the Hub metadata; the analysis compares the
   pairs ticked in the discovery's table (the server keeps the discovery). */

/** The job endpoints of this page. */
const DISCOVER_URL = '/api/forensics/lineage/discover';
const ANALYSE_URL = '/api/forensics/lineage';
const POLL_URL = '/api/forensics/lineage';

/** The input panels folded away while the pair comparison runs (and behind the finished summary). */
const LINEAGE_PANELS = ['input-panel', 'discovery-panel'];

/** The discovery the pair table was built from. */
let discoveryId = null;

/**
 * Fill a table body, hiding its block when there are no rows.
 * @param {string} bodyId
 * @param {string} blockId
 * @param {string[]} rows - HTML rows
 */
function fillTable(bodyId, blockId, rows) {
    const body = document.getElementById(bodyId);
    body.innerHTML = rows.join('');
    wireCopyButtons(body);
    document.getElementById(blockId).style.display = rows.length ? '' : 'none';
}

/**
 * Render the discovery: parents, children, comparable pairs (ticked) and skipped repositories.
 * @param {object} result - {discovery_id, discovery}
 */
function renderDiscovery(result) {
    const discovery = result.discovery;
    discoveryId = result.discovery_id;
    collapsePanels(LINEAGE_PANELS, false);   // a fresh discovery reopens the inputs collapsed by a prior run
    document.getElementById('discovery-target').textContent = `${discovery.target} (${discovery.kind})`;

    fillTable('parents-body', 'parents-block', discovery.parents.map((parent) => `<tr>
        <td>${escapeHtml(parent.relation)}</td>
        <td class="tech-id text-break">${escapeHtml(parent.repo)}</td>
        <td>${copyButton(parent.repo)}</td></tr>`));
    fillTable('children-body', 'children-block', discovery.children.map((child) => `<tr>
        <td class="tech-id text-break">${escapeHtml(child.repo)}</td>
        <td>${escapeHtml(child.relation)}</td>
        <td class="text-mono">${escapeHtml(formatNumber(child.downloads))}</td>
        <td>${copyButton(child.repo)}</td></tr>`));
    fillTable('skipped-body', 'skipped-block', discovery.skipped.map((item) => `<tr>
        <td class="tech-id text-break">${escapeHtml(item.repo)}</td>
        <td class="text-break">${escapeHtml(item.reason)}</td>
        <td>${copyButton(item.repo)}</td></tr>`));

    document.getElementById('pairs-body').innerHTML = discovery.pairs.length
        ? discovery.pairs.map((pair, index) => `<tr>
            <td><input type="checkbox" class="pair-check" data-index="${escapeAttribute(index)}" checked></td>
            <td class="tech-id text-break">${escapeHtml(pair.parent)}</td>
            <td class="tech-id text-break">${escapeHtml(pair.child)}</td>
            <td>${escapeHtml(pair.kind)}</td></tr>`).join('')
        : '<tr><td colspan="4" class="text-muted">No comparable pair was found.</td></tr>';
    document.getElementById('analyse-btn').style.display = discovery.pairs.length ? '' : 'none';

    renderMessages('discovery-notes-list', 'discovery-notes-block', discovery.notes);
    document.getElementById('analysis-panel').style.display = 'none';
    showSection('results');
}

/**
 * Render the finished comparison of the chosen pairs: the lineage summary only.
 * @param {object} result - {analysis, report, report_error}
 */
function renderAnalysis(result) {
    document.getElementById('analysis-panel').style.display = '';
    renderToolResult(result);
}

/**
 * Handle the discover button click.
 */
async function handleRun() {
    const model = document.getElementById('model-input').value.trim();
    if (!model) {
        notify('Enter a Hugging Face repository id', 'error');
    } else {
        const payload = {
            model,
            mode: document.getElementById('mode-select').value,
            max_variants: document.getElementById('variants-input').value,
            max_size_gb: document.getElementById('size-input').value,
        };
        const result = await runJob(DISCOVER_URL, POLL_URL, payload, 'Discovering relatives');
        if (result) renderDiscovery(result);
    }
}

/**
 * Handle the analyse button click: compare the ticked pairs.
 */
async function handleAnalyse() {
    const pairs = Array.from(document.querySelectorAll('.pair-check:checked')).map((box) => Number(box.dataset.index));
    if (!pairs.length) {
        notify('Tick at least one pair', 'error');
    } else {
        const result = await runJob(ANALYSE_URL, POLL_URL, { discovery_id: discoveryId, pairs }, 'Comparing the pairs',
            null, null, (collapsed) => collapsePanels(LINEAGE_PANELS, collapsed));
        if (result) renderAnalysis(result);
    }
}

/**
 * Reattach to the running or just-finished lineage job after a navigation.
 * The two steps share one kind, so the finished result's shape says which it was: a discovery
 * payload rebuilds the pair table; an analysis payload shows the lineage summary.
 * @param {object} job - the current-job payload
 */
async function reattach(job) {
    collapsePanels(LINEAGE_PANELS, true);
    const result = await followJob(POLL_URL, job.job_id, job.label);
    if (result && result.discovery) {
        renderDiscovery(result);
    } else if (result) {
        renderAnalysis(result);
    } else {
        collapsePanels(LINEAGE_PANELS, false);
    }
}

document.addEventListener('DOMContentLoaded', () => {
    new HubAutocomplete(document.getElementById('model-input'), document.getElementById('model-input-dropdown'));
    document.getElementById('run-btn').addEventListener('click', handleRun);
    initJobReattach('lineage', reattach);
    document.getElementById('analyse-btn').addEventListener('click', handleAnalyse);
    ['model-input', 'variants-input', 'size-input'].forEach((id) => {
        document.getElementById(id).addEventListener('keydown', (e) => {
            if (e.key === 'Enter') handleRun();
        });
    });
});
