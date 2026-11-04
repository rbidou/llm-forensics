/* LLM Forensics - Documentation */

/** Shorter queries match almost every chapter, so they are ignored. */
const MINIMUM_SEARCH_LENGTH = 2;

/** Characters that would otherwise be read as regular expression syntax. */
const REGEX_SPECIALS = /[.*+?^${}()|[\]\\]/g;

/**
 * Every chapter, with a pristine copy of the content it arrived with.
 * Restoring the copy undoes highlighting exactly, rather than by stripping
 * <mark> elements back out of edited markup.
 * @type {{section: HTMLDetailsElement, original: HTMLDetailsElement, entry: HTMLElement|null}[]}
 */
const chapters = [];

/**
 * The contents entry pointing at a chapter.
 * @param {string} anchor - the chapter's id
 * @returns {HTMLElement|null}
 */
function contentsEntryFor(anchor) {
    return document.querySelector(`#doc-contents-list li[data-target="${CSS.escape(anchor)}"]`);
}

/**
 * Wrap every occurrence of a query in <mark>.
 * Text nodes are walked rather than the markup searched and replaced: a
 * replacement over innerHTML would corrupt a tag whose name or attribute
 * contained the query, and would let the query itself inject markup.
 * @param {HTMLElement} root
 * @param {string} query
 * @returns {number} how many occurrences were marked
 */
function highlightWithin(root, query) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const needle = query.toLowerCase();
    const hits = [];

    while (walker.nextNode()) {
        if (walker.currentNode.nodeValue.toLowerCase().includes(needle)) hits.push(walker.currentNode);
    }

    let count = 0;
    const pattern = new RegExp(`(${query.replace(REGEX_SPECIALS, '\\$&')})`, 'gi');

    hits.forEach((node) => {
        const fragment = document.createDocumentFragment();
        node.nodeValue.split(pattern).forEach((piece, index) => {
            if (index % 2 === 1) {
                const mark = document.createElement('mark');
                mark.textContent = piece;
                fragment.append(mark);
                count += 1;
            } else if (piece) {
                fragment.append(document.createTextNode(piece));
            }
        });
        node.replaceWith(fragment);
    });

    return count;
}

/**
 * Restore every chapter to its unhighlighted, closed, visible state.
 */
function clearSearch() {
    chapters.forEach(({ section, original, entry }) => {
        section.replaceChildren(...original.cloneNode(true).childNodes);
        section.hidden = false;
        section.open = false;
        if (entry) entry.hidden = false;
    });
}

/**
 * Show only the chapters matching the query, with every occurrence marked.
 * @param {string} query
 */
function searchDocumentation(query) {
    clearSearch();

    const status = document.getElementById('doc-search-status');
    let matchedChapters = 0;
    let matches = 0;

    if (query.length >= MINIMUM_SEARCH_LENGTH) {
        const needle = query.toLowerCase();
        chapters.forEach(({ section, entry }) => {
            const found = section.textContent.toLowerCase().includes(needle) ? highlightWithin(section, query) : 0;
            section.hidden = found === 0;
            section.open = found > 0;
            if (entry) entry.hidden = found === 0;
            if (found > 0) {
                matchedChapters += 1;
                matches += found;
            }
        });
        status.textContent = matches
            ? `${matches} match${matches === 1 ? '' : 'es'} in ${matchedChapters} chapter${matchedChapters === 1 ? '' : 's'}`
            : 'No matches';
    } else {
        status.textContent = '';
    }
}

/**
 * Handle typing in the search box.
 * @param {Event} event
 */
function handleSearchInput(event) {
    searchDocumentation(event.target.value.trim());
}

/**
 * Handle a contents link: it points at a closed chapter, so open it on the way.
 * @param {Event} event
 */
function handleContentsClick(event) {
    const target = document.getElementById(event.currentTarget.getAttribute('href').slice(1));
    if (target) target.open = true;
}

/**
 * Index the chapters and wire the search box and the contents links.
 */
function initDocumentation() {
    const search = document.getElementById('doc-search');
    if (search) {
        document.querySelectorAll('.doc-section').forEach((section) => {
            chapters.push({ section, original: section.cloneNode(true), entry: contentsEntryFor(section.id) });
        });
        search.addEventListener('input', handleSearchInput);
        document.querySelectorAll('#doc-contents-list a').forEach((link) => {
            link.addEventListener('click', handleContentsClick);
        });
    }
}

document.addEventListener('DOMContentLoaded', initDocumentation);
