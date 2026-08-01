/**
 * PDFViewer — PDF.js integration with bbox-based chunk highlighting
 * Works with the DocRAG workspace layout
 */
pdfjsLib.GlobalWorkerOptions.workerSrc =
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js';

const PDFViewer = (() => {
  let pdfDoc        = null;
  let currentPage   = 1;
  let totalPages    = 0;
  let scale         = 1.5;
  let docId         = null;
  let rendering     = false;
  let pendingPage   = null;
  let renderTask    = null;
  let activeHighlights = [];         // [{chunk_id, page_number, bbox, page_width, page_height}]
  const pageCiteCounts = {};         // page → citation count (for heatmap)

  // DOM
  const canvas    = () => document.getElementById('pdfCanvas');
  const hlLayer   = () => document.getElementById('highlightLayer');
  const pdfWrap   = () => document.getElementById('pdfCanvasWrap');
  const pdfArea   = () => document.getElementById('pdfArea');
  const pageEl    = () => document.getElementById('currentPage');
  const totalEl   = () => document.getElementById('totalPages');
  const zoomEl    = () => document.getElementById('zoomLevel');

  // ── PUBLIC ──────────────────────────────────────────────────────

  async function loadDocument(url, dId) {
    docId = dId;
    try {
      pdfDoc = await pdfjsLib.getDocument(url).promise;
      totalPages = pdfDoc.numPages;
      totalEl().textContent = totalPages;
      currentPage = 1;
      await renderPage(1);
      updateNavBtns();
    } catch(e) {
      console.error('PDF load error:', e);
      showToast('Failed to load PDF viewer', 'error');
    }
  }

  async function goToPage(n) {
    if (!pdfDoc) return;
    n = Math.max(1, Math.min(n, totalPages));
    currentPage = n;
    await renderPage(n);
    updateNavBtns();
  }

  function setScale(s) {
    scale = Math.max(0.5, Math.min(3.0, s));
    zoomEl().textContent = Math.round(scale * 100) + '%';
    if (pdfDoc) renderPage(currentPage);
  }

  /** Highlight an array of chunk objects on the PDF */
  function highlightChunks(chunks) {
    if (!chunks || !chunks.length) { clearHighlights(); return; }
    activeHighlights = chunks;

    // Update heatmap counts
    chunks.forEach(c => {
      const p = c.page_number;
      pageCiteCounts[p] = (pageCiteCounts[p] || 0) + 1;
    });
    updateHeatmap();

    // Navigate to first cited page, then draw
    const firstPage = chunks[0].page_number;
    if (firstPage !== currentPage) {
      goToPage(firstPage).then(() => drawHighlights(chunks, firstPage));
    } else {
      drawHighlights(chunks, currentPage);
    }
  }

  function clearHighlights() {
    activeHighlights = [];
    const hl = hlLayer();
    if (hl) hl.innerHTML = '';
  }

  function getCurrentPage() { return currentPage; }
  function getTotalPages()   { return totalPages; }

  // ── RENDER ──────────────────────────────────────────────────────

  async function renderPage(n) {
    if (rendering) { pendingPage = n; return; }
    rendering = true;
    pageEl().textContent = n;

    try {
      const page = await pdfDoc.getPage(n);
      const vp   = page.getViewport({ scale });
      const c    = canvas();
      const ctx  = c.getContext('2d');

      c.width  = vp.width;
      c.height = vp.height;

      const wrap = pdfWrap();
      wrap.style.width  = vp.width  + 'px';
      wrap.style.height = vp.height + 'px';

      const hl = hlLayer();
      hl.style.width  = vp.width  + 'px';
      hl.style.height = vp.height + 'px';

      if (renderTask) { try { renderTask.cancel(); } catch(_){} }
      renderTask = page.render({ canvasContext: ctx, viewport: vp });
      await renderTask.promise;

      // Redraw any highlights for this page
      const pageHLs = activeHighlights.filter(h => h.page_number === n);
      if (pageHLs.length) drawHighlights(pageHLs, n);
      else hl.innerHTML = '';

    } catch(e) {
      if (e?.name !== 'RenderingCancelledException') console.error('Render error:', e);
    } finally {
      rendering = false;
      if (pendingPage !== null) {
        const p = pendingPage; pendingPage = null;
        renderPage(p);
      }
    }
  }

  // ── HIGHLIGHTS ──────────────────────────────────────────────────

  function drawHighlights(chunks, pageNum) {
    const hl = hlLayer();
    hl.innerHTML = '';
    const cw = canvas().width;
    const ch = canvas().height;

    const fallbackW = cw / scale;
    const fallbackH = ch / scale;

    chunks.forEach((chunk, i) => {
      if (chunk.page_number !== pageNum) return;
      const bbox   = chunk.bbox;
      const origW  = chunk.page_width || fallbackW;
      const origH  = chunk.page_height || fallbackH;
      if (!bbox) return;

      let x0, y0, x1, y1;
      if (Array.isArray(bbox) && bbox.length >= 4) {
          [x0, y0, x1, y1] = bbox;
      } else if (typeof bbox === 'object') {
          x0 = bbox.x0; y0 = bbox.y0; x1 = bbox.x1; y1 = bbox.y1;
      } else if (typeof bbox === 'string') {
          try {
             const parsed = JSON.parse(bbox);
             if (Array.isArray(parsed)) [x0, y0, x1, y1] = parsed;
             else { x0 = parsed.x0; y0 = parsed.y0; x1 = parsed.x1; y1 = parsed.y1; }
          } catch(e) { return; }
      }
      if (x0 === undefined || y0 === undefined) return;

      const sx = cw / origW;
      const sy = ch / origH;

      const div = document.createElement('div');
      div.className = 'hl-box';
      div.style.left   = (x0 * sx) + 'px';
      div.style.top    = (y0 * sy) + 'px';
      div.style.width  = ((x1 - x0) * sx) + 'px';
      div.style.height = ((y1 - y0) * sy) + 'px';
      // Enhance the highlight visibility according to project theme (light blue)
      div.style.backgroundColor = 'rgba(147, 205, 248, 0.4)';
      div.style.border = '2px solid #1f6187';
      div.style.animationDelay = (i * 0.08) + 's';
      div.title = chunk.chunk_id;
      hl.appendChild(div);
    });

    // Scroll first highlight into view
    const first = hl.querySelector('.hl-box');
    if (first) setTimeout(() => first.scrollIntoView({ behavior: 'smooth', block: 'center' }), 100);
  }

  // ── HEATMAP ──────────────────────────────────────────────────────

  function updateHeatmap() {
    const el = document.getElementById('heatmapBars');
    if (!el) return;
    el.innerHTML = '';

    const entries = Object.entries(pageCiteCounts)
      .sort((a, b) => b[1] - a[1]).slice(0, 12);
    if (!entries.length) return;

    const maxCount = entries[0][1];
    entries.forEach(([page, count]) => {
      const barH = Math.max(5, (count / maxCount) * 18);
      const wrap  = document.createElement('div');
      wrap.style.cssText = 'display:flex;flex-direction:column;align-items:center;gap:2px;cursor:pointer';
      wrap.title = `Page ${page}: ${count} citation(s)`;
      wrap.onclick = () => goToPage(parseInt(page));

      const fill = document.createElement('div');
      fill.style.cssText = `width:14px;height:${barH}px;background:rgba(31,97,135,.25);border:1px solid rgba(145,200,228,.4);border-radius:1px;transition:all .15s`;
      fill.onmouseenter = () => fill.style.background = 'rgba(31,97,135,.5)';
      fill.onmouseleave = () => fill.style.background = 'rgba(31,97,135,.25)';

      const label = document.createElement('div');
      label.style.cssText = 'font-family:"Space Grotesk",monospace;font-size:8px;color:rgba(65,72,78,.5)';
      label.textContent = 'p' + page;

      wrap.appendChild(fill);
      wrap.appendChild(label);
      el.appendChild(wrap);
    });
  }

  // ── MODES ────────────────────────────────────────────────────────

  async function showServerHighlight(pageNum, chunkIds) {
    if (!docId) return;
    const params = chunkIds.length ? `?chunks=${chunkIds.join(',')}` : '';
    const img = document.getElementById('hlImage');
    img.src = `/api/page-render/${docId}/${pageNum}${params}`;
    document.getElementById('pdfArea').classList.add('hidden');
    document.getElementById('hlImageArea').classList.remove('hidden');
  }

  function showInteractive() {
    document.getElementById('pdfArea').classList.remove('hidden');
    document.getElementById('hlImageArea').classList.add('hidden');
  }

  // ── NAV ──────────────────────────────────────────────────────────

  function updateNavBtns() {
    const prev = document.getElementById('prevPage');
    const next = document.getElementById('nextPage');
    if (prev) prev.disabled = currentPage <= 1;
    if (next) next.disabled = currentPage >= totalPages;
  }

  // ── EVENTS ───────────────────────────────────────────────────────

  document.getElementById('prevPage')?.addEventListener('click', () => {
    if (currentPage > 1) goToPage(currentPage - 1);
  });
  document.getElementById('nextPage')?.addEventListener('click', () => {
    if (currentPage < totalPages) goToPage(currentPage + 1);
  });
  document.getElementById('zoomIn')?.addEventListener('click',  () => setScale(scale + 0.25));
  document.getElementById('zoomOut')?.addEventListener('click', () => setScale(scale - 0.25));
  document.getElementById('fitBtn')?.addEventListener('click',  () => setScale(1.5));

  document.getElementById('modeInteractive')?.addEventListener('click', function() {
    this.className = 'font-mono text-[10px] px-2 py-1 border border-primary bg-primary text-on-primary transition-all';
    const hi = document.getElementById('modeHighlight');
    if (hi) hi.className = 'font-mono text-[10px] px-2 py-1 border border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary transition-all';
    showInteractive();
  });

  document.getElementById('modeHighlight')?.addEventListener('click', function() {
    this.className = 'font-mono text-[10px] px-2 py-1 border border-primary bg-primary text-on-primary transition-all';
    const pi = document.getElementById('modeInteractive');
    if (pi) pi.className = 'font-mono text-[10px] px-2 py-1 border border-outline-variant text-on-surface-variant hover:border-primary hover:text-primary transition-all';
    const chunkIds = activeHighlights.map(h => h.chunk_id);
    showServerHighlight(currentPage, chunkIds);
  });

  return { loadDocument, goToPage, setScale, highlightChunks, clearHighlights, getCurrentPage, getTotalPages, showInteractive };
})();