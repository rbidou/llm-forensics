/* LLM Forensics - Directory picker
   A folder input with a Browse navigator that selects directories only, for the
   Settings page. It walks this machine's folders through the same folder-listing
   endpoint as the model picker, but lists folders (never files) and its "Use this
   folder" button takes the folder currently being browsed. The markup comes from
   the directory_picker macro (templates/app/macros.html). */

/** The folder-listing endpoint backing the Browse navigator. */
const DIRECTORY_FOLDERS_URL = '/api/forensics/folders';

/**
 * A folder input and its directory-only Browse navigator.
 */
class DirectoryPicker {
    /**
     * @param {string} prefix - the id prefix of the macro's elements (e.g. 'reports-dir')
     */
    constructor(prefix) {
        this.prefix = prefix;
        this.input = document.getElementById(`${prefix}-input`);
        this.browser = document.getElementById(`${prefix}-browser`);
        this.rootSelect = document.getElementById(`${prefix}-browser-root`);
        this.up = document.getElementById(`${prefix}-browser-up`);
        this.pathLabel = document.getElementById(`${prefix}-browser-path`);
        this.error = document.getElementById(`${prefix}-browser-error`);
        this.body = document.getElementById(`${prefix}-browser-body`);
        this.listing = null;
        this.fallback = '';  // where to open when the field is empty (set by the page)

        document.getElementById(`${prefix}-browse-btn`).addEventListener('click', () => this.open());
        document.getElementById(`${prefix}-browser-choose`).addEventListener('click', () => this.choose());
        document.getElementById(`${prefix}-browser-cancel`).addEventListener('click', () => this.close());
        this.up.addEventListener('click', () => this.browseUp());
        this.rootSelect.addEventListener('change', () => this.browseTo(this.rootSelect.value));
    }

    /**
     * Open the navigator on the folder in the field, or on the fallback when it is empty.
     */
    open() {
        this.browser.style.display = '';
        this.browseTo(this.input.value.trim().replace(/^"|"$/g, '') || this.fallback);
    }

    /**
     * List a folder and render it; a folder that cannot be read shows its reason.
     * @param {string} path
     */
    async browseTo(path) {
        const response = await apiFetch(`${DIRECTORY_FOLDERS_URL}?path=${encodeURIComponent(path)}`);
        if (response.status) {
            this.listing = response.data;
            this.error.style.display = 'none';
            this.render(response.data);
        } else {
            this.error.textContent = response.message || 'This folder cannot be read';
            this.error.style.display = '';
        }
    }

    /**
     * Draw a folder listing: its drives, its path and its subfolders (files are not shown).
     * @param {object} listing - {path, parent, folders, files, roots}
     */
    render(listing) {
        this.rootSelect.innerHTML = listing.roots.map((root) =>
            `<option value="${escapeAttribute(root)}">${escapeHtml(root)}</option>`).join('');
        const current = listing.roots.find((root) => listing.path.startsWith(root));
        if (current) this.rootSelect.value = current;
        this.pathLabel.textContent = listing.path;
        this.up.disabled = !listing.parent;

        const folders = listing.folders;
        this.body.innerHTML = folders.length
            ? folders.map((item) =>
                `<tr class="browser-row"><td><span class="browser-kind">folder</span> ` +
                `${escapeHtml(item.name)}</td></tr>`).join('')
            : '<tr><td class="text-muted">No subfolders here.</td></tr>';
        this.body.querySelectorAll('.browser-row').forEach((row, index) => {
            row.addEventListener('click', () => this.browseTo(folders[index].path));
        });
    }

    /**
     * Move up to the parent folder, when there is one.
     */
    browseUp() {
        if (this.listing && this.listing.parent) this.browseTo(this.listing.parent);
    }

    /**
     * Take the folder being browsed as the value and close the navigator.
     */
    choose() {
        if (this.listing) {
            this.input.value = this.listing.path;
            this.close();
        }
    }

    /**
     * Hide the Browse navigator.
     */
    close() {
        this.browser.style.display = 'none';
    }
}
