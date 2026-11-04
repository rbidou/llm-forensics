/* LLM Forensics - Model picker
   One reference field with a Detect step, shared by every page that takes a
   model. Detection says what the reference is; the picker then shows only the
   choices it leaves open: the checkpoint of a training run, the file of a GGUF
   folder (most precise first), the base an adapter is applied to.

   The field also offers two ways to fill it: a Hub auto-completion that suggests
   repository ids as the user types one (HubAutocomplete), and a Browse button
   that walks this machine's folders and the GGUF files on it. The markup comes
   from the model_picker macro (templates/app/macros.html). */

/** The detection endpoint. */
const DETECT_URL = '/api/forensics/detect';

/** The Hub auto-completion endpoint (suggests repository ids). */
const HUB_SEARCH_URL = '/api/forensics/hub/search';

/** The folder-listing endpoint backing the Browse navigator. */
const FOLDERS_URL = '/api/forensics/folders';

/** Below this many characters a Hub query matches too much to be useful. */
const MINIMUM_HUB_QUERY = 2;

/** Milliseconds to wait after a keystroke before querying the Hub. */
const HUB_SEARCH_DELAY_MS = 250;

/**
 * Whether the text is clearly a filesystem path rather than a Hub id, so the
 * Hub is not searched while a path is being typed. Mirrors the engine's rule.
 * @param {string} text
 * @returns {boolean}
 */
function looksLikePath(text) {
    const slashes = text.split('/').length - 1;
    return '.~/\\'.includes(text.charAt(0)) || text.includes('\\') || slashes > 1;
}

/**
 * A short right-hand description of a Hub search hit (its download count, or its author).
 * @param {object} item - a hub/search result ({modelId, author, downloads, ...})
 * @returns {string}
 */
function hubMeta(item) {
    return item.downloads ? `${item.downloads} downloads` : (item.author || '');
}

/**
 * Auto-completion of a Hub repository id for a text input, with a dropdown and
 * keyboard navigation. Suggestions come from the server (the page may not call
 * the Hub itself), debounced, and only while the input looks like a Hub id.
 */
class HubAutocomplete {
    /**
     * @param {HTMLInputElement} input
     * @param {HTMLElement} dropdown - the empty .autocomplete-dropdown next to the input
     * @param {?Function} onPick - called with the chosen id after it is written to the input
     */
    constructor(input, dropdown, onPick = null) {
        this.input = input;
        this.dropdown = dropdown;
        this.onPick = onPick;
        this.items = [];
        this.selectedIndex = -1;
        this.timer = null;

        this.input.addEventListener('input', () => this.handleInput());
        this.input.addEventListener('keydown', (e) => this.handleKeydown(e));
        document.addEventListener('click', (e) => this.handleOutsideClick(e));
    }

    /**
     * Schedule a search a short while after the last keystroke, unless the input
     * is too short or looks like a path.
     */
    handleInput() {
        clearTimeout(this.timer);
        const query = this.input.value.trim();
        if (query.length < MINIMUM_HUB_QUERY || looksLikePath(query)) {
            this.hide();
            return;
        }
        this.timer = setTimeout(() => this.search(query), HUB_SEARCH_DELAY_MS);
    }

    /**
     * Fetch the suggestions for a query and show them.
     * @param {string} query
     */
    async search(query) {
        const response = await apiFetch(`${HUB_SEARCH_URL}?q=${encodeURIComponent(query)}`);
        this.items = response.status ? response.data.models : [];
        this.selectedIndex = -1;
        this.render();
    }

    /**
     * Draw the current suggestions, highlighting the selected one.
     */
    render() {
        if (this.items.length === 0) {
            this.hide();
            return;
        }
        this.dropdown.innerHTML = this.items.map((item, index) =>
            `<div class="autocomplete-item${index === this.selectedIndex ? ' selected' : ''}">` +
            `<span class="item-id">${escapeHtml(item.modelId)}</span>` +
            `<span class="item-meta">${escapeHtml(hubMeta(item))}</span></div>`).join('');
        this.dropdown.querySelectorAll('.autocomplete-item').forEach((el, index) => {
            el.addEventListener('mousedown', (e) => {
                e.preventDefault();
                this.choose(index);
            });
        });
        this.dropdown.classList.add('show');
    }

    /**
     * Move through the suggestions and accept or dismiss them with the keyboard.
     * @param {KeyboardEvent} e
     */
    handleKeydown(e) {
        if (!this.dropdown.classList.contains('show')) return;
        if (e.key === 'ArrowDown') {
            e.preventDefault();
            this.selectedIndex = Math.min(this.selectedIndex + 1, this.items.length - 1);
            this.render();
        } else if (e.key === 'ArrowUp') {
            e.preventDefault();
            this.selectedIndex = Math.max(this.selectedIndex - 1, 0);
            this.render();
        } else if (e.key === 'Enter' && this.selectedIndex >= 0) {
            e.preventDefault();
            this.choose(this.selectedIndex);
        } else if (e.key === 'Escape') {
            this.hide();
        }
    }

    /**
     * Adopt the suggestion at an index: write its id to the input and notify.
     * @param {number} index
     */
    choose(index) {
        const item = this.items[index];
        if (item) {
            this.input.value = item.modelId;
            this.hide();
            if (this.onPick) this.onPick(item.modelId);
        }
    }

    /**
     * Hide the dropdown and forget the suggestions.
     */
    hide() {
        this.dropdown.classList.remove('show');
        this.dropdown.innerHTML = '';
        this.items = [];
        this.selectedIndex = -1;
    }

    /**
     * Hide the dropdown when a click lands outside the input and the dropdown.
     * @param {MouseEvent} e
     */
    handleOutsideClick(e) {
        if (!this.input.contains(e.target) && !this.dropdown.contains(e.target)) this.hide();
    }
}

/**
 * A model reference input and its follow-up choices.
 */
class ModelPicker {
    /**
     * @param {string} prefix - the id prefix of the macro's elements (e.g. 'base')
     * @param {?Function} onDetect - called with the detection payload after a successful detect
     */
    constructor(prefix, onDetect = null) {
        this.prefix = prefix;
        this.onDetect = onDetect;
        this.input = document.getElementById(`${prefix}-ref`);
        this.info = document.getElementById(`${prefix}-info`);
        this.checkpointGroup = document.getElementById(`${prefix}-checkpoint-group`);
        this.checkpointSelect = document.getElementById(`${prefix}-checkpoint`);
        this.ggufGroup = document.getElementById(`${prefix}-gguf-group`);
        this.ggufSelect = document.getElementById(`${prefix}-gguf`);
        this.baseGroup = document.getElementById(`${prefix}-base-group`);
        this.baseInput = document.getElementById(`${prefix}-base-override`);
        this.dropdown = document.getElementById(`${prefix}-dropdown`);
        this.browser = document.getElementById(`${prefix}-browser`);
        this.browserRoot = document.getElementById(`${prefix}-browser-root`);
        this.browserUp = document.getElementById(`${prefix}-browser-up`);
        this.browserPath = document.getElementById(`${prefix}-browser-path`);
        this.browserError = document.getElementById(`${prefix}-browser-error`);
        this.browserBody = document.getElementById(`${prefix}-browser-body`);
        this.detected = null;
        this.listing = null;
        this.autocomplete = new HubAutocomplete(this.input, this.dropdown, () => this.reset());

        document.getElementById(`${prefix}-detect-btn`).addEventListener('click', () => this.handleDetect());
        document.getElementById(`${prefix}-browse-btn`).addEventListener('click', () => this.handleBrowse());
        document.getElementById(`${prefix}-browser-choose`).addEventListener('click', () => this.chooseFolder());
        document.getElementById(`${prefix}-browser-cancel`).addEventListener('click', () => this.closeBrowser());
        this.browserUp.addEventListener('click', () => this.browseUp());
        this.browserRoot.addEventListener('change', () => this.browseTo(this.browserRoot.value));
        this.input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !this.dropdown.classList.contains('show')) this.handleDetect();
        });
        this.input.addEventListener('input', () => this.reset());
        this.checkpointSelect.addEventListener('change', () => this.handleCheckpoint());
    }

    /**
     * Forget the previous detection (the reference changed).
     */
    reset() {
        this.detected = null;
        this.info.textContent = '';
        this.info.classList.remove('error');
        [this.checkpointGroup, this.ggufGroup, this.baseGroup].forEach((group) => {
            group.style.display = 'none';
        });
    }

    /**
     * Detect the typed reference and show the choices it leaves open.
     * @returns {Promise<boolean>} whether the reference is usable as it stands
     */
    async handleDetect() {
        const ref = this.input.value.trim().replace(/^"|"$/g, '');
        let usable = false;
        this.reset();
        if (!ref) {
            notify('Enter a local path or a Hugging Face repository id', 'error');
        } else {
            this.info.textContent = 'Detecting…';
            const response = await apiFetch(DETECT_URL, { method: 'POST', body: { ref } });
            if (response.status) {
                this.detected = response.data;
                this.render(response.data);
                usable = response.data.kind !== null;
                if (this.onDetect) this.onDetect(response.data);
            } else {
                this.info.textContent = response.message || 'Detection failed';
                this.info.classList.add('error');
            }
        }
        return usable;
    }

    /**
     * Show the detection result and the controls it calls for.
     * @param {object} data - the detection payload
     */
    render(data) {
        if (data.checkpoints.length) {
            this.info.textContent = `Training run: pick one of its ${data.checkpoints.length} checkpoints.`;
            this.checkpointSelect.innerHTML = '<option value="">Choose a checkpoint…</option>' +
                data.checkpoints.map((item) =>
                    `<option value="${escapeAttribute(item.path)}">${escapeHtml(item.name)}</option>`).join('');
            this.checkpointGroup.style.display = '';
        } else {
            this.renderModel(data);
        }
    }

    /**
     * Show a detected model and the GGUF / adapter-base controls it calls for.
     * @param {object} data - the detection payload of a usable model
     */
    renderModel(data) {
        const details = [data.label];
        if (data.base_model) details.push(`declared base ${data.base_model}`);
        this.info.textContent = details.join(' · ');

        if (data.gguf_files.length > 1) {
            this.ggufSelect.innerHTML = data.gguf_files.map((item) =>
                `<option value="${escapeAttribute(item.path)}">${escapeHtml(item.name)}` +
                `${item.most_precise ? ' (most precise)' : ''}</option>`).join('');
            this.ggufGroup.style.display = '';
        }
        if (data.kind === 'adapter' && this.baseInput) {
            this.baseInput.placeholder = data.base_model || 'The adapter declares no base: enter one';
            this.baseGroup.style.display = '';
        }
    }

    /**
     * Use the chosen checkpoint as the reference, and detect it.
     */
    async handleCheckpoint() {
        const path = this.checkpointSelect.value;
        if (path) {
            this.input.value = path;
            await this.handleDetect();
        }
    }

    /**
     * Open the Browse navigator on the folder currently in the field.
     */
    handleBrowse() {
        this.autocomplete.hide();
        this.browser.style.display = '';
        this.browseTo(this.input.value.trim().replace(/^"|"$/g, ''));
    }

    /**
     * List a folder and render it; a folder that cannot be read shows its reason.
     * @param {string} path
     */
    async browseTo(path) {
        const response = await apiFetch(`${FOLDERS_URL}?path=${encodeURIComponent(path)}`);
        if (response.status) {
            this.listing = response.data;
            this.browserError.style.display = 'none';
            this.renderBrowser(response.data);
        } else {
            this.browserError.textContent = response.message || 'This folder cannot be read';
            this.browserError.style.display = '';
        }
    }

    /**
     * Draw a folder listing: its drives, its path, and its subfolders and GGUF files.
     * @param {object} listing - {path, parent, folders, files, roots}
     */
    renderBrowser(listing) {
        this.browserRoot.innerHTML = listing.roots.map((root) =>
            `<option value="${escapeAttribute(root)}">${escapeHtml(root)}</option>`).join('');
        const current = listing.roots.find((root) => listing.path.startsWith(root));
        if (current) this.browserRoot.value = current;
        this.browserPath.textContent = listing.path;
        this.browserUp.disabled = !listing.parent;

        const entries = [
            ...listing.folders.map((item) => ({ name: item.name, path: item.path, kind: 'folder' })),
            ...listing.files.map((item) => ({ name: item.name, path: item.path, kind: 'file' })),
        ];
        this.browserBody.innerHTML = entries.length
            ? entries.map((entry) =>
                `<tr class="browser-row"><td><span class="browser-kind">` +
                `${escapeHtml(entry.kind === 'folder' ? 'folder' : 'gguf')}</span> ` +
                `${escapeHtml(entry.name)}</td></tr>`).join('')
            : '<tr><td class="text-muted">No subfolders or GGUF files here.</td></tr>';
        this.browserBody.querySelectorAll('.browser-row').forEach((row, index) => {
            row.addEventListener('click', () => this.openEntry(entries[index]));
        });
    }

    /**
     * Descend into a folder, or select a GGUF file as the reference.
     * @param {object} entry - {name, path, kind}
     */
    openEntry(entry) {
        if (entry.kind === 'folder') {
            this.browseTo(entry.path);
        } else {
            this.useReference(entry.path);
        }
    }

    /**
     * Move up to the parent folder, when there is one.
     */
    browseUp() {
        if (this.listing && this.listing.parent) this.browseTo(this.listing.parent);
    }

    /**
     * Take the folder being browsed as the reference.
     */
    chooseFolder() {
        if (this.listing) this.useReference(this.listing.path);
    }

    /**
     * Put a browsed path in the field, close the navigator and detect it.
     * @param {string} path
     */
    useReference(path) {
        this.closeBrowser();
        this.input.value = path;
        this.reset();
        this.handleDetect();
    }

    /**
     * Hide the Browse navigator.
     */
    closeBrowser() {
        this.browser.style.display = 'none';
    }

    /**
     * Detect the reference unless it already was.
     * @returns {Promise<boolean>} whether the reference is usable
     */
    async ensureDetected() {
        const ready = this.detected !== null && this.detected.kind !== null;
        return ready || this.handleDetect();
    }

    /**
     * The model choice sent to the API.
     * @returns {object} {ref, gguf_file, base_override}
     */
    value() {
        const gguf = this.ggufGroup.style.display === '' ? this.ggufSelect.value : '';
        const base = this.baseInput && this.baseGroup.style.display === '' ? this.baseInput.value.trim() : '';
        return { ref: this.input.value.trim().replace(/^"|"$/g, ''), gguf_file: gguf || null, base_override: base || null };
    }

    /**
     * A one-line description of the choice, for confirmations and history.
     * @returns {string}
     */
    describe() {
        const choice = this.value();
        return [choice.ref, choice.gguf_file, choice.base_override ? `base ${choice.base_override}` : '']
            .filter(Boolean).join(' · ');
    }
}

/**
 * Detect every picker that is not detected yet.
 * @param {ModelPicker[]} pickers
 * @returns {Promise<boolean>} whether all of them are usable
 */
async function ensureAllDetected(pickers) {
    let usable = true;
    for (const picker of pickers) {
        usable = (await picker.ensureDetected()) && usable;
    }
    if (!usable) notify('Settle every model reference first (see the messages under the fields)', 'error');
    return usable;
}
