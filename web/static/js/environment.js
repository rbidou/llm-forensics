/* LLM Forensics - Environment */

/** The environment endpoint. */
const ENVIRONMENT_URL = '/api/forensics/environment';

/**
 * A label / value row.
 * @param {string} label
 * @param {string} value
 * @returns {string} HTML
 */
function valueRow(label, value) {
    return `<tr>
        <td>${escapeHtml(label)}</td>
        <td class="text-mono text-break">${escapeHtml(value)}</td>
    </tr>`;
}

/**
 * Render the environment report.
 * @param {object} data - the environment payload
 */
function renderResults(data) {
    const gpu = data.gpu
        ? `${data.gpu.name}, ${data.gpu.total_memory_mb} MiB, driver ${data.gpu.driver_version}`
        : 'none detected (CPU only)';
    const system = document.getElementById('system-body');
    system.innerHTML = [
        valueRow('Version', `LLM Forensics ${data.version}`),
        valueRow('Platform', data.platform),
        valueRow('Python', `${data.python} (${data.executable})`),
        valueRow('GPU', gpu),
        valueRow('CUDA (torch)', data.cuda),
    ].join('');

    document.getElementById('features-body').innerHTML = data.features.map((feature) => `<tr>
        <td>${escapeHtml(feature.label)}</td>
        <td>${feature.available
            ? '<span class="badge badge-success">yes</span>'
            : '<span class="badge badge-danger">no</span>'}</td>
        <td class="text-muted">${escapeHtml(feature.reason)}</td>
    </tr>`).join('');

    const packages = document.getElementById('packages-body');
    packages.innerHTML = data.packages.map((item) =>
        valueRow(item.name, item.version || 'not installed')).join('');
    showSection('results');
}

/**
 * Observe the machine again and render the report.
 */
async function handleRefresh() {
    showSection('loading');
    const response = await apiFetch(ENVIRONMENT_URL);
    if (response.status) {
        renderResults(response.data);
    } else {
        showError(response.message || 'The environment could not be observed.');
    }
}

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('refresh-btn').addEventListener('click', handleRefresh);
    handleRefresh();
});
