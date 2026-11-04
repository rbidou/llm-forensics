/* LLM Forensics - Reports list controller, shared by the Tools and Analysis report pages.
   Both list /api/forensics/reports; each page passes initReportsPage a filter choosing which
   analyses it shows (the per-tool reports, or the workflow assessments) and the row renderer
   for its own columns (toolsReportRow / analysisReportRow). */

/** The reports endpoint (GET lists, DELETE <id> removes one). */
const REPORTS_URL = '/api/forensics/reports';

/** The analysis name of a workflow report (the rest are per-tool reports). */
const ASSESSMENT_ANALYSIS = 'assessment';

/** Input keys naming the analysed model, most specific first (see engine.py meta.inputs). */
const MODEL_KEYS = ['tuned', 'suspect', 'model'];

/** Assessment risk level -> badge class (mirrors reports/html.py RISK_CLASSES). */
const RISK_BADGE = { clean: 'badge-success', low: 'badge-neutral', elevated: 'badge-warning', high: 'badge-danger' };

/** Keeps the reports this page shows (set by initReportsPage). */
let reportsFilter = () => true;

/** Builds one table row for this page (set by initReportsPage). */
let reportRow = toolsReportRow;

/** The report waiting for the delete confirmation. */
let pendingDelete = null;

/**
 * The display name of the analysed model of a report, read from its inputs. Comparison tools
 * record {base, tuned}, provenance and assessment {suspect}, lineage {model}; the analysed model
 * is the tuned / suspect / model one, falling back to the last input when none of those is present.
 * A Hub id (``org/model``) is kept whole; a local path is shortened to its file name.
 * @param {object} report - a listed report ({inputs})
 * @returns {string} the model name, or '' when the inputs name none
 */
function analysedModel(report) {
    const inputs = report.inputs || {};
    const key = MODEL_KEYS.find((name) => inputs[name]);
    const values = Object.values(inputs);
    let ref = '';
    if (key) {
        ref = inputs[key];
    } else if (values.length) {
        ref = values[values.length - 1];
    }
    return shortModelName(ref);
}

/**
 * A model reference as shown in the Model column (mirrors store.py _short_name): the last path
 * component of a local path, a Hub id (``org/model``) kept whole. A value is treated as a path
 * when it is absolute, holds a backslash, or starts with a dot.
 * @param {string} ref - a model reference (Hub id or local path)
 * @returns {string} the file name of a path, otherwise the reference unchanged
 */
function shortModelName(ref) {
    const text = String(ref || '');
    const isPath = /^([a-zA-Z]:[\\/]|[\\/])/.test(text) || text.includes('\\') || text.startsWith('.');
    const parts = text.replace(/\\/g, '/').replace(/\/+$/, '').split('/');
    return isPath ? parts[parts.length - 1] : text;
}

/**
 * A saved report's ISO timestamp as a human-readable local date and time.
 * @param {string} iso - an ISO 8601 timestamp (the result's meta.created)
 * @returns {string} the localised date-time, or the raw value when it cannot be parsed
 */
function formatTimestamp(iso) {
    const date = new Date(iso);
    return Number.isNaN(date.getTime()) ? String(iso || '') : date.toLocaleString();
}

/**
 * The action buttons of a report row: view, download, and delete. Each button carries both a text
 * label and an icon; the stylesheet shows the label on a wide screen and the icon below 1400px.
 * @param {object} report - a listed report with its links
 * @returns {string} HTML (every interpolated URL is attribute-escaped)
 */
function reportActions(report) {
    return `<td><div class="button-row action-row">
        <a class="btn btn-sm action-btn" href="${escapeAttribute(report.view_url)}" target="_blank" rel="noopener noreferrer" title="View" aria-label="View">
            <span class="action-text">View</span><span class="action-icon action-icon-view" aria-hidden="true"></span></a>
        <a class="btn btn-sm action-btn" href="${escapeAttribute(report.html_url)}" download title="HTML" aria-label="HTML">
            <span class="action-text">HTML</span><span class="action-icon action-icon-html" aria-hidden="true"></span></a>
        <a class="btn btn-sm action-btn" href="${escapeAttribute(report.json_url)}" download title="JSON" aria-label="JSON">
            <span class="action-text">JSON</span><span class="action-icon action-icon-json" aria-hidden="true"></span></a>
        <button class="btn btn-sm btn-danger action-btn delete-btn" data-id="${escapeAttribute(report.id)}" title="Delete" aria-label="Delete">
            <span class="action-text">Delete</span><span class="action-icon action-icon-delete" aria-hidden="true"></span></button>
    </div></td>`;
}

/**
 * One row of the Tools reports table: Model, Analysis, Created, Size, Actions.
 * @param {object} report - a listed per-tool report
 * @returns {string} HTML
 */
function toolsReportRow(report) {
    return `<tr>
        <td class="text-break">${escapeHtml(analysedModel(report))}</td>
        <td><span class="badge badge-accent">${escapeHtml(report.analysis)}</span></td>
        <td class="text-mono">${escapeHtml(formatTimestamp(report.created))}</td>
        <td class="text-mono text-nowrap">${escapeHtml(formatSize(report.size_bytes))}</td>
        ${reportActions(report)}
    </tr>`;
}

/**
 * One row of the Analysis reports table: Model, Created, Risk, Size, Actions.
 * @param {object} report - a listed workflow assessment
 * @returns {string} HTML
 */
function analysisReportRow(report) {
    const badgeClass = RISK_BADGE[report.risk] || 'badge-neutral';
    const badge = report.risk ? `<span class="badge ${badgeClass}">${escapeHtml(report.risk)}</span>` : '';
    return `<tr>
        <td class="text-break">${escapeHtml(analysedModel(report))}</td>
        <td>${escapeHtml(report.format)}</td>
        <td class="text-mono">${escapeHtml(formatTimestamp(report.created))}</td>
        <td>${badge}</td>
        <td class="text-mono text-nowrap">${escapeHtml(formatSize(report.size_bytes))}</td>
        ${reportActions(report)}
    </tr>`;
}

/**
 * Render the list of reports this page shows.
 * @param {object} data - {reports, root}
 */
function renderResults(data) {
    const reports = data.reports.filter(reportsFilter);
    document.getElementById('reports-root').textContent = data.root;
    document.getElementById('reports-empty').style.display = reports.length ? 'none' : '';
    document.getElementById('reports-table').style.display = reports.length ? '' : 'none';
    const body = document.getElementById('reports-body');
    body.innerHTML = reports.map(reportRow).join('');
    body.querySelectorAll('.delete-btn').forEach((btn) => {
        btn.addEventListener('click', () => openDeleteModal(btn.dataset.id));
    });
    showSection('results');
}

/**
 * Load the list of reports.
 */
async function handleRefresh() {
    showSection('loading');
    const response = await apiFetch(REPORTS_URL);
    if (response.status) {
        renderResults(response.data);
    } else {
        showError(response.message || 'The reports could not be listed.');
    }
}

/**
 * Ask for confirmation before deleting a report.
 * @param {string} reportId
 */
function openDeleteModal(reportId) {
    pendingDelete = reportId;
    document.getElementById('delete-target').textContent = reportId;
    document.getElementById('delete-overlay').style.display = 'flex';
}

/**
 * Close the confirmation without deleting anything.
 */
function closeDeleteModal() {
    pendingDelete = null;
    document.getElementById('delete-overlay').style.display = 'none';
}

/**
 * Delete the confirmed report, then refresh the list.
 */
async function handleDelete() {
    const reportId = pendingDelete;
    closeDeleteModal();
    if (reportId) {
        const response = await apiFetch(`${REPORTS_URL}/${encodeURIComponent(reportId)}`, { method: 'DELETE' });
        if (response.status) {
            notify(`Deleted ${reportId}`, 'success');
            await handleRefresh();
        } else {
            showError(response.message || 'The report could not be deleted.');
        }
    }
}

/**
 * Wire the page and load its reports.
 * @param {Function} filter - keeps the reports this page shows (a listed report -> boolean)
 * @param {Function} rowRenderer - builds a table row for this page (a listed report -> HTML)
 */
function initReportsPage(filter, rowRenderer) {
    reportsFilter = filter;
    reportRow = rowRenderer;
    document.getElementById('refresh-btn').addEventListener('click', handleRefresh);
    document.getElementById('delete-close').addEventListener('click', closeDeleteModal);
    document.getElementById('delete-cancel').addEventListener('click', closeDeleteModal);
    document.getElementById('delete-confirm').addEventListener('click', handleDelete);
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') closeDeleteModal();
    });
    handleRefresh();
}
