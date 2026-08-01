/**
 * EvidencePanel
 * - Renders answers with CLICKABLE citation chips (no raw HTML shown)
 * - Manages sidebar chunk cards
 * - Manages evidence history panel
 * - Coordinates with PDFViewer for highlights
 */

const EvidencePanel = (() => {

  let currentChunks = [];   // chunks from the most recent query
  let activeCardIdx  = -1;

  // Evidence history: [{query, rewrittenQuery, chunks, confidence, timestamp}]
  const historyLog = [];

  // ── PUBLIC API ────────────────────────────────────────────────

  /** Called after every successful RAG response */
  function setChunks(chunks, queryText, rewrittenQuery, confidence) {
    currentChunks = chunks || [];

    // Update sidebar top-k panel
    renderChunksSidebar(currentChunks);

    // Log to history
    if (currentChunks.length > 0) {
      historyLog.unshift({ query: queryText, rewrittenQuery, chunks: currentChunks, confidence, timestamp: Date.now() });
      renderHistorySidebar();
    }

    // Auto-highlight in PDF
    PDFViewer.highlightChunks(currentChunks);
  }

  function clear() {
    currentChunks = [];
    activeCardIdx = -1;
    renderChunksSidebar([]);
    PDFViewer.clearHighlights();
  }

  /** Handle click on a [pN:cN] chip in the chat */
  function handleCitationClick(event, chunkId, chunkIndex, pageNum) {
    event.preventDefault();
    PDFViewer.goToPage(parseInt(pageNum));

    const chunk = currentChunks[chunkIndex] || currentChunks.find(c => c.chunk_id === chunkId);
    if (chunk) PDFViewer.highlightChunks([chunk]);

    // Highlight matching card in sidebar
    activateCard(chunkIndex);

    // Switch to chunks tab
    const chunksTab = document.querySelector('[data-tab="tab-chunks"]');
    if (chunksTab) chunksTab.click();

    return false;
  }

  /**
   * Parse answer text → rich HTML with marked.js + clickable citation chips.
   * Strategy:
   *  1. Replace [pN:cN] tokens with unique placeholder strings so marked doesn't mangle them.
   *  2. Run the text through marked.js for full markdown rendering.
   *  3. Set innerHTML on a wrapper div (marked output is HTML).
   *  4. Walk the DOM and replace placeholder text nodes with real <a> chip elements.
   */
  function renderAnswerHTML(answerText, chunks) {
    if (!answerText) return document.createTextNode('');

    // Configure marked for safe, pretty output
    if (typeof marked !== 'undefined') {
      marked.setOptions({
        breaks: true,       // \n → <br>
        gfm: true,          // GitHub Flavored Markdown (tables, strikethrough)
        mangle: false,
        headerIds: false
      });
    }

    // --- Step 1: Extract citations & replace with unique placeholders ---
    const citMap = {};   // placeholder → {chunkId, chunkPos, pageNum}
    let cidx = 0;
    const withPlaceholders = answerText.replace(/\[p(\d+):c(\d+)\]/g, (match, pg, ch) => {
      const key = `CITPLACEHOLDER${cidx++}END`;
      const chunkId  = `p${pg}:c${ch}`;
      const chunkPos = chunks.findIndex(c => c.chunk_id === chunkId);
      citMap[key] = { chunkId, chunkPos, pageNum: parseInt(pg) };
      return key;
    });

    // --- Step 2: Render markdown ---
    let html = withPlaceholders;
    if (typeof marked !== 'undefined') {
      try { html = marked.parse(withPlaceholders); } catch(e) { /* fallback to plain */ }
    } else {
      // Minimal fallback: just convert newlines to <br>
      html = withPlaceholders.replace(/\n/g, '<br>');
    }

    // --- Step 3: Set innerHTML on a scoped wrapper ---
    const wrapper = document.createElement('div');
    wrapper.className = 'md-content';
    wrapper.innerHTML = html;

    // --- Step 4: Walk text nodes and replace placeholders with chip elements ---
    if (Object.keys(citMap).length > 0) {
      _replacePlaceholders(wrapper, citMap, chunks);
    }

    return wrapper;
  }

  /** Recursively walk all text nodes and replace CITPLACEHOLDER...END with chip <a> elements */
  function _replacePlaceholders(node, citMap, chunks) {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent;
      const keys = Object.keys(citMap);
      for (const key of keys) {
        if (text.includes(key)) {
          const parts = text.split(key);
          const frag = document.createDocumentFragment();
          parts.forEach((part, i) => {
            if (part) frag.appendChild(document.createTextNode(part));
            if (i < parts.length - 1) {
              const { chunkId, chunkPos, pageNum } = citMap[key];
              const chip = document.createElement('a');
              chip.className = 'cit-chip';
              chip.href = '#';
              chip.textContent = `[${chunkId}]`;
              chip.title = `Page ${pageNum} · Click to jump`;
              chip.addEventListener('click', e => {
                e.preventDefault();
                handleCitationClick(e, chunkId, chunkPos, pageNum);
              });
              frag.appendChild(chip);
            }
          });
          node.parentNode.replaceChild(frag, node);
          return; // node replaced, stop recursing this branch
        }
      }
    } else if (node.nodeType === Node.ELEMENT_NODE) {
      // Clone children list since we modify the DOM during iteration
      [...node.childNodes].forEach(child => _replacePlaceholders(child, citMap, chunks));
    }
  }

  // ── SIDEBAR: TOP-K CHUNKS ────────────────────────────────────

  function renderChunksSidebar(chunks) {
    const el = document.getElementById('chunksList');
    if (!el) return;
    el.innerHTML = '';

    if (!chunks.length) {
      el.innerHTML = '<div class="text-center py-8 px-4"><div class="font-mono text-xs text-on-surface-variant/50 leading-relaxed">Ask a question to see retrieved chunks here.</div></div>';
      return;
    }

    chunks.forEach((chunk, i) => {
      const card = buildChunkCard(chunk, i);
      el.appendChild(card);
    });
  }

  function buildChunkCard(chunk, index) {
    const card = document.createElement('div');
    card.className = 'border border-outline-variant bg-surface hover:border-primary hover:bg-surface-container-low transition-all cursor-pointer p-2.5 mb-1.5';
    card.dataset.index = index;

    const chunkId  = chunk.chunk_id    || '—';
    const pageNum  = chunk.page_number || '?';
    const preview  = (chunk.text_preview || chunk.text || '').slice(0, 120) + '…';
    const hybrid   = fmtScore(chunk.hybrid_score);
    const dense    = fmtScore(chunk.dense_score);
    const bm25     = fmtScore(chunk.bm25_score);
    const rerank   = fmtScore(chunk.rerank_score);
    const method   = chunk.retrieval_method || 'hybrid';
    const ctype    = chunk.content_type    || 'text';

    card.innerHTML = `
      <div class="flex items-center justify-between mb-1.5">
        <span class="font-mono text-xs font-semibold text-primary">${esc(chunkId)}</span>
        <span class="font-mono text-[10px] bg-primary/10 text-primary px-1.5 py-0.5">${hybrid}</span>
      </div>
      <div class="flex gap-1 mb-1.5 flex-wrap">
        <span class="font-mono text-[9px] bg-primary/5 border border-primary/20 text-primary/70 px-1.5 py-0.5">D:${dense}</span>
        <span class="font-mono text-[9px] bg-green-50 border border-green-200 text-green-700 px-1.5 py-0.5">B:${bm25}</span>
        <span class="font-mono text-[9px] bg-blue-50 border border-blue-200 text-blue-600 px-1.5 py-0.5">R:${rerank}</span>
        <span class="font-mono text-[9px] bg-surface-container border border-outline-variant text-on-surface-variant/60 px-1.5 py-0.5">${esc(method)}</span>
        <span class="font-mono text-[9px] bg-surface-container border border-outline-variant text-on-surface-variant/60 px-1.5 py-0.5">${esc(ctype)}</span>
      </div>
      <p class="text-xs text-on-surface-variant leading-relaxed mb-1.5">${esc(preview)}</p>
      <div class="flex items-center justify-between">
        <span class="font-mono text-[10px] text-on-surface-variant/50">Page ${pageNum}</span>
        <div class="flex gap-1">
          <button class="font-mono text-[9px] text-primary hover:underline jump-btn" data-page="${pageNum}" data-index="${index}">→ p${pageNum}</button>
          <button class="font-mono text-[9px] text-primary hover:underline hl-btn" data-index="${index}">◈ Highlight</button>
        </div>
      </div>`;

    // Card click → jump + highlight
    card.addEventListener('click', e => {
      if (e.target.closest('.jump-btn') || e.target.closest('.hl-btn')) return;
      activateCard(index, card);
      PDFViewer.goToPage(pageNum);
      PDFViewer.highlightChunks([chunk]);
    });

    card.querySelector('.jump-btn')?.addEventListener('click', e => {
      e.stopPropagation();
      activateCard(index, card);
      PDFViewer.goToPage(parseInt(card.querySelector('.jump-btn').dataset.page));
    });

    card.querySelector('.hl-btn')?.addEventListener('click', e => {
      e.stopPropagation();
      activateCard(index, card);
      PDFViewer.highlightChunks([chunk]);
      PDFViewer.goToPage(chunk.page_number);
    });

    return card;
  }

  function activateCard(index, cardEl) {
    document.querySelectorAll('#chunksList > div').forEach(c => {
      c.style.borderColor = '';
      c.style.background  = '';
    });
    const target = cardEl || document.querySelector(`#chunksList > div[data-index="${index}"]`);
    if (target) {
      target.style.borderColor = '#1f6187';
      target.style.background  = 'rgba(31,97,135,.06)';
      target.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
    activeCardIdx = index;
  }

  // ── SIDEBAR: EVIDENCE HISTORY ─────────────────────────────────

  function renderHistorySidebar() {
    const el = document.getElementById('evidenceList');
    if (!el) return;
    el.innerHTML = '';

    if (!historyLog.length) {
      el.innerHTML = '<div class="text-center py-8 px-4"><div class="font-mono text-xs text-on-surface-variant/50 leading-relaxed">No queries yet.</div></div>';
      return;
    }

    historyLog.forEach((entry, i) => {
      const card = buildHistoryCard(entry, i);
      el.appendChild(card);
    });
  }

  function buildHistoryCard(entry, idx) {
    const conf      = entry.confidence || 0;
    const confClass = conf >= 0.7 ? 'text-green-700 bg-green-50 border-green-200'
                    : conf >= 0.4 ? 'text-amber-700 bg-amber-50 border-amber-200'
                    :               'text-red-700 bg-red-50 border-red-200';
    const confLabel = conf >= 0.7 ? 'HIGH' : conf >= 0.4 ? 'MED' : 'LOW';
    const timeStr   = new Date(entry.timestamp).toLocaleTimeString([], {hour:'2-digit',minute:'2-digit'});

    const card = document.createElement('div');
    card.className = 'border border-outline-variant bg-surface hover:border-primary hover:bg-surface-container-low transition-all cursor-pointer p-2.5 mb-1.5';

    const rewriteNote = (entry.rewrittenQuery && entry.rewrittenQuery !== entry.query)
      ? `<div class="font-mono text-[9px] text-on-surface-variant/50 mt-1 italic">↺ ${esc(entry.rewrittenQuery.slice(0, 60))}…</div>`
      : '';

    card.innerHTML = `
      <div class="text-xs font-semibold text-on-surface mb-1 leading-snug">${esc(entry.query.slice(0, 80))}${entry.query.length > 80 ? '…' : ''}</div>
      ${rewriteNote}
      <div class="flex items-center gap-2 mt-2 flex-wrap">
        <span class="font-mono text-[9px] bg-primary/10 border border-primary/20 text-primary px-1.5 py-0.5">${entry.chunks.length} src</span>
        <span class="font-mono text-[9px] border px-1.5 py-0.5 ${confClass}">${confLabel} ${(conf*100).toFixed(0)}%</span>
        <span class="font-mono text-[9px] text-on-surface-variant/40 ml-auto">${timeStr}</span>
      </div>`;

    // Click → restore chunks for that query
    card.addEventListener('click', () => {
      setChunks(entry.chunks, entry.query, entry.rewrittenQuery, entry.confidence);
      // Switch to chunks tab
      const tab = document.querySelector('[data-tab="tab-chunks"]');
      if (tab) tab.click();
    });

    return card;
  }

  // ── HELPERS ───────────────────────────────────────────────────

  function esc(str) {
    return String(str || '')
      .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
      .replace(/"/g,'&quot;');
  }

  function fmtScore(n) {
    return typeof n === 'number' ? n.toFixed(3) : '—';
  }

  return {
    setChunks,
    clear,
    renderAnswerHTML,
    handleCitationClick,
    renderChunksSidebar,
  };

})();