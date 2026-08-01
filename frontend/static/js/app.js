/**
 * app.js — Main DocRAG workspace logic
 * Handles: upload, ingestion polling, chat, session, document list
 * All data fetched dynamically — no hardcoding
 */

// ── STATE ──────────────────────────────────────────────────────────
const APP = {
  docId: null,
  filename: null,
  sessionId: genSessionId(),
  loading: false,
  pageCount: 0,
};

// ── INIT ───────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('sessionInfo').textContent = 'sess:' + APP.sessionId.slice(0, 8);
  loadDocumentList();         // check for already-indexed docs
  wireEvents();
});

function wireEvents() {
  // File inputs
  document.getElementById('fileInputMain')?.addEventListener('change', e => handleFile(e.target.files[0]));
  document.getElementById('fileInputNav')?.addEventListener('change', e => handleFile(e.target.files[0]));

  // Drag & drop on upload zone
  const zone = document.getElementById('dropZone');
  zone?.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('border-primary', 'bg-surface-container'); });
  zone?.addEventListener('dragleave', () => zone.classList.remove('border-primary', 'bg-surface-container'));
  zone?.addEventListener('drop', e => {
    e.preventDefault();
    zone.classList.remove('border-primary', 'bg-surface-container');
    const f = e.dataTransfer.files[0];
    if (f?.name.endsWith('.pdf')) handleFile(f);
    else showToast('Please upload a PDF file', 'error');
  });

  // Chat input
  document.getElementById('queryInput')?.addEventListener('keydown', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendQuery(); }
  });
  document.getElementById('queryInput')?.addEventListener('input', onInputChange);
  document.getElementById('sendBtn')?.addEventListener('click', sendQuery);

  // Clear chat buttons
  document.getElementById('clearChatBtn')?.addEventListener('click', clearConversation);
  document.getElementById('clearChatBtn2')?.addEventListener('click', clearConversation);
}

// ── TAB SWITCHING (sidebar) ────────────────────────────────────────
function switchTab(tabId, btnEl) {
  document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(tabId)?.classList.add('active');
  btnEl?.classList.add('active');
}

// ── DOCUMENT LIST ──────────────────────────────────────────────────
async function loadDocumentList() {
  try {
    const res = await fetch('/api/documents');
    const data = await res.json();
    const docs = data.documents || [];

    renderDocList(docs);

    // Show existing docs on upload screen if any exist
    if (docs.length > 0) {
      const hint = document.getElementById('existingDocsHint');
      const list = document.getElementById('existingDocsList');
      if (hint && list) {
        hint.classList.remove('hidden');
        list.innerHTML = '';
        docs.forEach(doc => {
          const btn = document.createElement('button');
          btn.className = 'w-full flex items-center gap-2 border border-outline-variant hover:border-primary hover:bg-surface-container-low transition-all p-2 text-left';
          btn.innerHTML = `
            <span class="text-primary text-sm">📄</span>
            <div class="flex-1 min-w-0">
              <div class="text-xs font-semibold text-on-surface truncate">${esc(doc.filename || doc.doc_id)}</div>
              <div class="font-mono text-[10px] text-on-surface-variant/60">${doc.vector_count || 0} chunks indexed</div>
            </div>
            <span class="font-mono text-[10px] text-primary">Open →</span>`;
          btn.addEventListener('click', () => openExistingDoc(doc));
          list.appendChild(btn);
        });
      }
    }
  } catch (e) {
    console.error('Failed to load documents:', e);
  }
}

function renderDocList(docs) {
  const el = document.getElementById('docList');
  if (!el) return;

  if (!docs.length) {
    el.innerHTML = '<div class="text-center py-8 px-4"><div class="text-2xl mb-3">📄</div><div class="font-mono text-xs text-on-surface-variant/50 leading-relaxed">No documents indexed yet.<br/>Upload a PDF to begin.</div></div>';
    return;
  }

  el.innerHTML = '';
  docs.forEach(doc => {
    const isActive = doc.doc_id === APP.docId;
    const card = document.createElement('div');
    card.className = `flex items-center gap-2 p-2 mb-1.5 border cursor-pointer transition-all hover:bg-surface-container-low ${isActive ? 'border-primary bg-surface-container-low' : 'border-outline-variant bg-surface'}`;
    card.innerHTML = `
      <div class="w-8 h-8 flex items-center justify-center flex-shrink-0 ${isActive ? 'bg-primary text-on-primary' : 'bg-surface-container-high text-primary'} text-sm">📄</div>
      <div class="flex-1 min-w-0">
        <div class="text-xs font-semibold text-on-surface truncate">${esc(doc.filename || doc.doc_id)}</div>
        <div class="font-mono text-[10px] text-on-surface-variant/60">${doc.vector_count || 0} chunks · ${doc.total_pages || 0}pp</div>
      </div>
      <button class="text-red-400 hover:text-red-600 transition-colors text-sm opacity-0 group-hover:opacity-100 del-btn" data-doc-id="${esc(doc.doc_id)}" title="Delete">✕</button>`;

    card.classList.add('group');
    card.addEventListener('click', e => {
      if (e.target.closest('.del-btn')) return;
      openExistingDoc(doc);
    });
    card.querySelector('.del-btn')?.addEventListener('click', async e => {
      e.stopPropagation();
      if (!confirm(`Delete "${doc.filename || doc.doc_id}" index?`)) return;
      await deleteDoc(doc.doc_id);
    });

    el.appendChild(card);
  });
}

function openExistingDoc(doc) {
  APP.docId = doc.doc_id;
  APP.filename = doc.filename || doc.doc_id;
  APP.pageCount = doc.total_pages || 0;

  // Hide overlay, show workspace
  document.getElementById('uploadOverlay').style.display = 'none';
  document.getElementById('appShell').style.display = 'grid';

  setNavStatus('ready', APP.filename);
  PDFViewer.loadDocument(`/api/pdf/${APP.docId}`, APP.docId);
  renderDocList([doc]);  // refresh doc list active state
  showToast(`Opened: ${APP.filename}`, 'success');
}

async function deleteDoc(docId) {
  try {
    const res = await fetch(`/api/documents/${docId}`, { method: 'DELETE' });
    const d = await res.json();
    if (d.success) {
      showToast('Document deleted', 'info');
      if (APP.docId === docId) {
        APP.docId = null;
        APP.filename = null;
        setNavStatus('idle', 'No document loaded');
        document.getElementById('appShell').style.display = 'none';
        document.getElementById('uploadOverlay').style.display = 'flex';
      }
      loadDocumentList();
    }
  } catch (e) {
    showToast('Delete failed', 'error');
  }
}

// ── FILE UPLOAD ────────────────────────────────────────────────────
async function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.pdf')) { showToast('Only PDF files supported', 'error'); return; }

  // Show progress UI
  document.getElementById('uploadProgress').classList.remove('hidden');
  document.getElementById('existingDocsHint')?.classList.add('hidden');
  const dropZone = document.getElementById('dropZone');
  if (dropZone) dropZone.style.display = 'none';

  setStep(1, 'active'); setProgressFill(5);
  setProgressMsg('Uploading…');
  setNavStatus('loading', `Uploading ${file.name}…`);

  const fd = new FormData();
  fd.append('file', file);

  try {
    const res = await fetch('/api/upload', { method: 'POST', body: fd });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Upload failed');

    APP.docId = data.doc_id;
    APP.filename = data.filename;

    showToast(`Uploaded: ${file.name}`, 'success');
    pollIngestion(data.doc_id);
  } catch (e) {
    showToast(`Upload failed: ${e.message}`, 'error');
    setNavStatus('error', 'Upload failed');
    resetUploadUI();
  }
}

function pollIngestion(docId) {
  const src = new EventSource(`/api/ingestion-stream/${docId}`);

  src.onmessage = e => {
    const st = JSON.parse(e.data);
    setProgressFill(st.progress || 0);
    setProgressMsg(st.message || '');

    if (st.progress >= 10) setStep(1, 'active');
    if (st.progress >= 50) { setStep(1, 'done'); setStep(2, 'active'); }
    if (st.progress >= 70) { setStep(2, 'done'); setStep(3, 'active'); }
    if (st.progress >= 90) { setStep(3, 'done'); setStep(4, 'active'); }

    if (st.status === 'complete') {
      src.close();
      setStep(1, 'done'); setStep(2, 'done'); setStep(3, 'done'); setStep(4, 'done');
      setProgressFill(100);
      APP.pageCount = st.total_pages || 0;
      setTimeout(() => {
        document.getElementById('uploadOverlay').style.display = 'none';
        document.getElementById('appShell').style.display = 'grid';
        setNavStatus('ready', APP.filename);
        PDFViewer.loadDocument(`/api/pdf/${docId}`, docId);
        loadDocumentList();
        showToast(`Ready! ${st.total_pages || 0} pages indexed.`, 'success');
      }, 500);
    }

    if (st.status === 'error') {
      src.close();
      showToast(`Ingestion failed: ${st.error || 'unknown error'}`, 'error');
      setNavStatus('error', 'Ingestion failed');
      resetUploadUI();
    }
  };

  src.onerror = () => src.close();
}

// ── CHAT ───────────────────────────────────────────────────────────
async function sendQuery() {
  const input = document.getElementById('queryInput');
  const query = input?.value.trim();
  if (!query || APP.loading) return;
  if (!APP.docId) { showToast('Please open a document first', 'error'); return; }
  if (query.length > 500) { showToast('Query too long (max 500 chars)', 'error'); return; }

  appendUserMsg(query);
  input.value = '';
  input.style.height = 'auto';
  document.getElementById('charCount').textContent = '0 / 500';

  const thinkId = appendThinking();
  setLoading(true);

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query, doc_id: APP.docId, session_id: APP.sessionId })
    });
    const data = await res.json();
    removeMsg(thinkId);

    if (!res.ok || data.error) { appendErrorMsg(data.error || 'Request failed'); return; }

    const chunks = data.debug?.retrieved_chunks || [];
    appendAssistantMsg(data, query, chunks);

    // Update evidence panel with retrieved chunks
    EvidencePanel.setChunks(
      chunks,
      query,
      data.rewritten_query || query,
      data.confidence || 0
    );

  } catch (e) {
    removeMsg(thinkId);
    appendErrorMsg(`Network error: ${e.message}`);
  } finally {
    setLoading(false);
  }
}

async function clearConversation() {
  await fetch('/api/clear-conversation', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ session_id: APP.sessionId })
  });
  const msgs = document.getElementById('messages');
  msgs.innerHTML = `
    <div class="flex justify-center">
      <div class="bg-surface-container border border-outline-variant text-on-surface-variant text-xs font-mono px-4 py-2 max-w-sm text-center leading-relaxed">
        Conversation cleared. Ask a new question.
      </div>
    </div>`;
  EvidencePanel.clear();
  PDFViewer.clearHighlights();
  showToast('Conversation cleared', 'info');
}

// ── MESSAGE BUILDERS ───────────────────────────────────────────────
function appendUserMsg(text) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'flex justify-end msg-in';
  const bubble = document.createElement('div');
  bubble.className = 'max-w-[85%] bg-primary text-on-primary text-sm px-4 py-2.5 leading-relaxed';
  bubble.style.borderRadius = '10px 10px 2px 10px';
  bubble.textContent = text;
  div.appendChild(bubble);
  msgs.appendChild(div);
  scrollBottom();
}

function appendAssistantMsg(data, originalQuery, chunks) {
  const msgs = document.getElementById('messages');
  const outer = document.createElement('div');
  outer.className = 'flex flex-col gap-1.5 msg-in';

  // Build bubble using DOM (not innerHTML) to avoid raw HTML display
  const bubbleWrap = document.createElement('div');
  bubbleWrap.className = 'flex justify-start';

  const bubble = document.createElement('div');
  bubble.className = 'max-w-[90%] bg-surface border border-outline-variant text-on-surface text-sm px-4 py-3 leading-relaxed';
  bubble.style.borderRadius = '2px 10px 10px 10px';

  const isNF = data.not_found || (data.answer || '').includes('Not found in the document');

  if (isNF) {
    const tag = document.createElement('div');
    tag.className = 'flex items-center gap-2 font-mono text-xs text-red-700 bg-red-50 border border-red-200 px-3 py-2';
    tag.innerHTML = '✕ Not found in the document.';
    bubble.appendChild(tag);
  } else {
    // Use EvidencePanel.renderAnswerHTML to properly parse citations
    const rendered = EvidencePanel.renderAnswerHTML(data.answer || '', chunks);
    bubble.appendChild(rendered);
  }

  bubbleWrap.appendChild(bubble);
  outer.appendChild(bubbleWrap);

  // Footer row: confidence + sources toggle + rewrite note
  const footer = document.createElement('div');
  footer.className = 'flex items-center gap-2 flex-wrap px-1';

  const conf = data.confidence || 0;
  if (!isNF) {
    const confClass = conf >= 0.7
      ? 'font-mono text-[10px] bg-green-50 border border-green-200 text-green-700 px-2 py-0.5'
      : conf >= 0.4
        ? 'font-mono text-[10px] bg-amber-50 border border-amber-200 text-amber-700 px-2 py-0.5'
        : 'font-mono text-[10px] bg-red-50 border border-red-200 text-red-700 px-2 py-0.5';
    const confBadge = document.createElement('span');
    confBadge.className = confClass;
    confBadge.textContent = `conf: ${(conf * 100).toFixed(0)}%`;
    footer.appendChild(confBadge);
  }

  if (chunks.length > 0) {
    const srcBtn = document.createElement('button');
    srcBtn.className = 'font-mono text-[10px] text-primary hover:text-primary/70 flex items-center gap-1 transition-colors';
    srcBtn.textContent = `◈ ${chunks.length} source${chunks.length > 1 ? 's' : ''} retrieved`;
    srcBtn.addEventListener('click', () => {
      // Switch sidebar to chunks tab
      const tab = document.querySelector('[data-tab="tab-chunks"]');
      if (tab) tab.click();
      EvidencePanel.setChunks(chunks, originalQuery, data.rewritten_query || originalQuery, conf);
    });
    footer.appendChild(srcBtn);
  }

  if (data.rewritten_query && data.rewritten_query !== originalQuery) {
    const rw = document.createElement('span');
    rw.className = 'font-mono text-[10px] text-on-surface-variant/50 italic';
    rw.textContent = `↺ "${data.rewritten_query.slice(0, 50)}${data.rewritten_query.length > 50 ? '…' : ''}"`;
    footer.appendChild(rw);
  }

  outer.appendChild(footer);
  msgs.appendChild(outer);
  scrollBottom();
}

function appendThinking() {
  const msgs = document.getElementById('messages');
  const id = 'thinking-' + Date.now();
  const div = document.createElement('div');
  div.className = 'flex justify-start msg-in';
  div.id = id;
  const bubble = document.createElement('div');
  bubble.className = 'bg-surface border border-outline-variant px-4 py-3 text-on-surface-variant text-sm';
  bubble.style.borderRadius = '2px 10px 10px 10px';
  bubble.innerHTML = '<span class="blink" style="font-size:20px;color:#91C8E4">·</span><span class="blink" style="font-size:20px;color:#91C8E4;animation-delay:.2s">·</span><span class="blink" style="font-size:20px;color:#91C8E4;animation-delay:.4s">·</span>';
  div.appendChild(bubble);
  msgs.appendChild(div);
  scrollBottom();
  return id;
}

function appendErrorMsg(msg) {
  const msgs = document.getElementById('messages');
  const div = document.createElement('div');
  div.className = 'flex justify-start msg-in';
  const bubble = document.createElement('div');
  bubble.className = 'max-w-[85%] bg-red-50 border border-red-200 text-red-700 text-sm px-4 py-2.5';
  bubble.style.borderRadius = '2px 10px 10px 10px';
  bubble.textContent = '✕ ' + msg;
  div.appendChild(bubble);
  msgs.appendChild(div);
  scrollBottom();
}

function removeMsg(id) {
  document.getElementById(id)?.remove();
}

// ── INPUT HANDLING ─────────────────────────────────────────────────
function onInputChange() {
  const input = document.getElementById('queryInput');
  const count = document.getElementById('charCount');
  if (!input || !count) return;
  input.style.height = 'auto';
  input.style.height = Math.min(input.scrollHeight, 112) + 'px';
  const len = input.value.length;
  count.textContent = `${len} / 500`;
  count.style.color = len > 450 ? '#ba1a1a' : '';
}

// ── NAV STATUS ─────────────────────────────────────────────────────
function setNavStatus(type, text) {
  const dot = document.getElementById('statusDot');
  const label = document.getElementById('docStatusText');
  if (!dot || !label) return;

  const colors = { idle: 'rgba(255,255,255,.25)', loading: '#91C8E4', ready: '#4ade80', error: '#f87171' };
  dot.style.background = colors[type] || colors.idle;

  if (type === 'loading') dot.style.animation = 'pd 1.5s infinite';
  else dot.style.animation = '';

  label.textContent = text;
}

// ── PROGRESS HELPERS ───────────────────────────────────────────────
function setProgressFill(pct) {
  const el = document.getElementById('progressFill');
  if (el) el.style.width = pct + '%';
}

function setProgressMsg(msg) {
  const el = document.getElementById('progressMsg');
  if (el) el.textContent = msg;
}

function setStep(num, state) {
  const el = document.getElementById('step' + num);
  if (!el) return;
  const labels = ['', 'Extract', 'Chunk', 'Embed', 'Index'];
  if (state === 'active') {
    el.textContent = `◉ ${labels[num]}`;
    el.style.color = '#1f6187';
  } else if (state === 'done') {
    el.textContent = `✓ ${labels[num]}`;
    el.style.color = '#15803d';
  } else {
    el.textContent = `◯ ${labels[num]}`;
    el.style.color = '';
  }
}

function resetUploadUI() {
  document.getElementById('uploadProgress')?.classList.add('hidden');
  const dz = document.getElementById('dropZone');
  if (dz) dz.style.display = '';
  setProgressFill(0);
  [1, 2, 3, 4].forEach(n => setStep(n, 'idle'));
}

// ── TOAST ──────────────────────────────────────────────────────────
function showToast(msg, type = 'info') {
  const wrap = document.getElementById('toastWrap');
  if (!wrap) return;

  const borderColors = { success: '#15803d', error: '#ba1a1a', info: '#1f6187' };
  const icons = { success: '✓', error: '✕', info: 'ℹ' };

  const t = document.createElement('div');
  t.style.cssText = `display:flex;align-items:center;gap:8px;background:#1d1c0b;color:#fefadd;padding:9px 14px;font-family:'Space Grotesk',monospace;font-size:12px;border-left:3px solid ${borderColors[type] || borderColors.info};box-shadow:0 4px 20px rgba(0,0,0,.3);animation:msgIn .2s ease;min-width:200px;max-width:300px`;
  t.innerHTML = `<span>${icons[type] || 'ℹ'}</span><span>${esc(msg)}</span>`;
  wrap.appendChild(t);
  setTimeout(() => { t.style.opacity = '0'; t.style.transition = 'opacity .3s'; setTimeout(() => t.remove(), 300); }, 3500);
}
// expose globally for pdf_viewer.js
window.showToast = showToast;

// ── UTILITIES ──────────────────────────────────────────────────────
function scrollBottom() {
  const msgs = document.getElementById('messages');
  if (msgs) msgs.scrollTop = msgs.scrollHeight;
}

function setLoading(val) {
  APP.loading = val;
  const btn = document.getElementById('sendBtn');
  const input = document.getElementById('queryInput');
  if (btn) btn.disabled = val;
  if (input) input.disabled = val;
}

function genSessionId() {
  return 'sess_' + Math.random().toString(36).slice(2, 11);
}

function esc(str) {
  return String(str || '')
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}