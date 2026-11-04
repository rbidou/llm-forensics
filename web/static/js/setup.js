/* LLM Forensics - Settings */

/** The settings endpoints. */
const SETUP_URL = '/api/forensics/setup';
const SERVER_URL = '/api/forensics/setup/server';
const DIRECTORIES_URL = '/api/forensics/setup/directories';

/** The Browse navigators of the two directory fields (created on load). */
let reportsPicker = null;
let modelsPicker = null;

/**
 * Show the stored web server address in the form.
 * @param {object} server - the server part of the setup payload
 */
function renderServer(server) {
    document.getElementById('host-input').value = server.host;
    document.getElementById('port-input').value = server.port;
    document.getElementById('server-default-hint').textContent = `Default: ${server.defaultHost}:${server.defaultPort}`;
    document.getElementById('server-default-badge').style.display = server.isDefault ? '' : 'none';
    document.getElementById('server-exposed-warning').style.display = server.isLoopback ? 'none' : '';
    document.getElementById('server-reset-btn').disabled = server.isDefault;
}

/**
 * Show the stored report and models directories in the form.
 * @param {object} directories - the directories part of the setup payload
 */
function renderDirectories(directories) {
    document.getElementById('reports-dir-input').value = directories.reportsIsDefault ? '' : directories.reportsDir;
    document.getElementById('models-dir-input').value = directories.modelsDir;
    document.getElementById('reports-dir-default-hint').textContent = `Default: ${directories.defaultReportsDir}`;
    document.getElementById('models-dir-default-hint').textContent =
        `Working directory when empty: ${directories.workingDir}`;
    // Where each Browse navigator opens when its field is empty.
    if (reportsPicker) reportsPicker.fallback = directories.defaultReportsDir;
    if (modelsPicker) modelsPicker.fallback = directories.workingDir;
    const isDefault = directories.reportsIsDefault && directories.modelsIsDefault;
    document.getElementById('directories-default-badge').style.display = isDefault ? '' : 'none';
    document.getElementById('directories-reset-btn').disabled = isDefault;
}

/**
 * Load the current settings.
 */
async function loadSettings() {
    const response = await apiFetch(SETUP_URL);

    if (response.status) {
        renderServer(response.data.server);
        renderDirectories(response.data.directories);
    } else {
        showError(response.message || 'The settings could not be loaded.');
    }
}

/**
 * Store the web server address; empty values restore the defaults.
 * @param {string} host
 * @param {string} port
 */
async function saveServerAddress(host, port) {
    const response = await apiFetch(SERVER_URL, { method: 'POST', body: { host, port } });

    if (response.status) {
        showSection('none');
        renderServer(response.data.server);
        notify(response.message, 'success');
    } else {
        showError(response.message || 'The web server address could not be changed.');
    }
}

/**
 * Handle the server save button click.
 */
function handleServerSave() {
    saveServerAddress(
        document.getElementById('host-input').value.trim(),
        document.getElementById('port-input').value.trim(),
    );
}

/**
 * Store the report and models directories; empty values restore the defaults.
 * @param {string} reportsDir
 * @param {string} modelsDir
 */
async function saveDirectories(reportsDir, modelsDir) {
    const response = await apiFetch(DIRECTORIES_URL, { method: 'POST', body: { reports_dir: reportsDir, models_dir: modelsDir } });

    if (response.status) {
        showSection('none');
        renderDirectories(response.data.directories);
        notify(response.message, 'success');
    } else {
        showError(response.message || 'The directories could not be changed.');
    }
}

/**
 * Handle the directories save button click.
 */
function handleDirectoriesSave() {
    saveDirectories(
        document.getElementById('reports-dir-input').value.trim(),
        document.getElementById('models-dir-input').value.trim(),
    );
}

document.addEventListener('DOMContentLoaded', () => {
    reportsPicker = new DirectoryPicker('reports-dir');
    modelsPicker = new DirectoryPicker('models-dir');
    document.getElementById('server-save-btn').addEventListener('click', handleServerSave);
    document.getElementById('server-reset-btn').addEventListener('click', () => saveServerAddress('', ''));
    ['host-input', 'port-input'].forEach((id) => {
        document.getElementById(id).addEventListener('keydown', (event) => {
            if (event.key === 'Enter') handleServerSave();
        });
    });
    document.getElementById('directories-save-btn').addEventListener('click', handleDirectoriesSave);
    document.getElementById('directories-reset-btn').addEventListener('click', () => saveDirectories('', ''));
    ['reports-dir-input', 'models-dir-input'].forEach((id) => {
        document.getElementById(id).addEventListener('keydown', (event) => {
            if (event.key === 'Enter') handleDirectoriesSave();
        });
    });
    loadSettings();
});
