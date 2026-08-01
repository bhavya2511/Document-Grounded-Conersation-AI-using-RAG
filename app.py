import os
# CRITICAL: Resolve Windows DLL conflicts and Disable unstable Paddle 3.x features
# These MUST be set before any other imports
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["FLAGS_use_mkldnn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_new_executor"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_new_executor_serial_run"] = "1"

import json
import uuid
import logging
import threading
from functools import wraps
from pathlib import Path
from flask import (
    Flask, request, jsonify, render_template,
    send_file, session, Response, stream_with_context
)
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Import DB
from core.db import (
    create_user, authenticate_user, get_document, list_workspace_documents,
    create_document, update_document_status, delete_document,
    get_conversation, save_conversation, clear_conversation, get_db
)

load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Create required directories
REQUIRED_DIRS = [
    "./storage/qdrant_data",
    "./storage/uploads",
    "./storage/bm25_index"
]
for d in REQUIRED_DIRS:
    Path(d).mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder="frontend/templates",
    static_folder="frontend/static"
)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-production")
# Session cookie expires when browser is closed (session-only cookie)
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = os.getenv("UPLOAD_FOLDER", "./storage/uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# No upload size limit
app.config["MAX_CONTENT_LENGTH"] = None

ALLOWED_EXTENSIONS = {"pdf"}

# In-memory store strictly for live progress percentage tracking
ingestion_status_live = {} 

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def get_doc_id(filename):
    name = Path(filename).stem
    clean = "".join(c if c.isalnum() else "_" for c in name)
    return clean[:50]

# --- AUTH DECORATOR ---
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated_function


# ─────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────

@app.route("/")
def landing():
    return render_template("landing.html")

@app.route("/workspace")
def workspace():
    return render_template("index.html")

# ─────────────────────────────────────────────
# AUTH ENDPOINTS
# ─────────────────────────────────────────────

@app.route("/api/auth/signup", methods=["POST"])
def signup():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    try:
        user = create_user(email, password)
        session["user_id"] = user["user_id"]
        session["email"] = user["email"]
        session["workspace_id"] = user["default_workspace_id"]
        session.permanent = False
        return jsonify({"success": True, "user": user})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/auth/login", methods=["POST"])
def login():
    data = request.get_json()
    email = data.get("email")
    password = data.get("password")
    if not email or not password:
        return jsonify({"error": "Email and password required"}), 400
    
    user = authenticate_user(email, password)
    if not user:
        return jsonify({"error": "Invalid credentials"}), 401
        
    session["user_id"] = user["user_id"]
    session["email"] = user["email"]
    session["workspace_id"] = user["workspace_id"]
    session.permanent = False
    return jsonify({"success": True, "user": user})

@app.route("/api/auth/logout", methods=["POST"])
def logout():
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/me", methods=["GET"])
def get_me():
    if "user_id" in session:
        return jsonify({
            "user_id": session["user_id"],
            "email": session["email"],
            "workspace_id": session["workspace_id"]
        })
    return jsonify({"error": "Not logged in"}), 401

# ─────────────────────────────────────────────
# DOCUMENT UPLOAD + INGESTION
# ─────────────────────────────────────────────

@app.route("/api/upload", methods=["POST"])
@login_required
def upload_document():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported"}), 400

    filename = secure_filename(file.filename)
    doc_id = get_doc_id(filename)
    save_path = os.path.join(UPLOAD_FOLDER, filename)
    workspace_id = session["workspace_id"]

    # Ingestion guard: Check if already in DB and fully indexed
    existing_doc = get_document(doc_id)
    if existing_doc and existing_doc["workspace_id"] == workspace_id:
        if existing_doc["status"] == "indexed":
            return jsonify({
                "success": True,
                "doc_id": doc_id,
                "filename": filename,
                "already_indexed": True,
                "message": "Document already indexed. Skipping re-ingestion."
            })
    
    # Save file
    file.save(save_path)
    logger.info(f"Uploaded: {filename} → {save_path}")

    # Create document in MongoDB
    if not existing_doc:
        create_document(doc_id, workspace_id, session["user_id"], filename)
    else:
        update_document_status(doc_id, "uploaded")

    # Initialize live ingestion status
    ingestion_status_live[doc_id] = {
        "status": "processing",
        "progress": 0,
        "message": "Starting ingestion...",
        "total_pages": 0,
        "total_chunks": 0,
        "error": None
    }

    thread = threading.Thread(
        target=_run_ingestion_background,
        args=(save_path, doc_id, filename, workspace_id)
    )
    thread.daemon = True
    thread.start()

    return jsonify({
        "success": True,
        "doc_id": doc_id,
        "filename": filename,
        "already_indexed": False,
        "message": "Upload successful. Ingestion started."
    })


def _run_ingestion_background(pdf_path: str, doc_id: str, filename: str, workspace_id: str):
    try:
        from core.multimodal.extractor import extract_pdf_multimodal
        from core.ingestion.chunker import chunk_pages
        from core.ingestion.indexer import index_chunks
        
        update_document_status(doc_id, "processing")

        ingestion_status_live[doc_id]["message"] = "Analyzing document layout (Multimodal)..."
        ingestion_status_live[doc_id]["progress"] = 10

        # Try multimodal extraction (PaddleOCR + Vision) with automatic fallback
        pages = extract_pdf_multimodal(pdf_path)

        ingestion_status_live[doc_id]["total_pages"] = len(pages)
        ingestion_status_live[doc_id]["message"] = f"Extracted {len(pages)} pages. Chunking..."
        ingestion_status_live[doc_id]["progress"] = 50

        chunks = chunk_pages(pages, source_file=filename)

        ingestion_status_live[doc_id]["total_chunks"] = len(chunks)
        ingestion_status_live[doc_id]["message"] = f"Created {len(chunks)} chunks. Building index..."
        ingestion_status_live[doc_id]["progress"] = 70

        # Pass workspace_id to indexer
        stats = index_chunks(chunks, doc_id, workspace_id)

        update_document_status(doc_id, "indexed", {
            "total_pages": len(pages),
            "total_chunks": stats["indexed"]
        })

        ingestion_status_live[doc_id].update({
            "status": "complete",
            "progress": 100,
            "message": f"Ready. {stats['indexed']} chunks indexed.",
            "total_chunks": stats["indexed"],
            "total_pages": len(pages)
        })

        logger.info(f"Ingestion complete for {doc_id}: {stats}")

    except Exception as e:
        logger.error(f"Ingestion failed for {doc_id}: {e}", exc_info=True)
        update_document_status(doc_id, "error")
        ingestion_status_live[doc_id].update({
            "status": "error",
            "progress": 0,
            "message": "Ingestion failed.",
            "error": str(e)
        })

@app.route("/api/ingestion-status/<doc_id>")
@login_required
def get_ingestion_status(doc_id):
    # Try DB first
    doc = get_document(doc_id)
    if not doc or doc["workspace_id"] != session["workspace_id"]:
        return jsonify({"error": "Document not found"}), 404
        
    if doc["status"] == "indexed":
        return jsonify({
            "status": "complete",
            "progress": 100,
            "message": "Ready.",
            "total_pages": doc.get("total_pages", 0)
        })
        
    status = ingestion_status_live.get(doc_id, {
        "status": doc["status"],
        "progress": 0,
        "message": f"Status: {doc['status']}"
    })
    return jsonify(status)

@app.route("/api/ingestion-stream/<doc_id>")
@login_required
def ingestion_stream(doc_id):
    def generate():
        import time
        while True:
            # Check DB to see if it's already done
            doc = get_document(doc_id)
            if doc and doc["status"] == "indexed":
                yield f"data: {json.dumps({'status': 'complete', 'progress': 100, 'message': 'Ready.'})}\n\n"
                break
                
            status = ingestion_status_live.get(doc_id, {})
            data = json.dumps(status)
            yield f"data: {data}\n\n"

            if status.get("status") in ("complete", "error"):
                break
            time.sleep(1)

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )

# ─────────────────────────────────────────────
# CHAT / Q&A
# ─────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
@login_required
def chat():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    query = data.get("query", "").strip()
    doc_id = data.get("doc_id", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))
    workspace_id = session["workspace_id"]

    if not query or not doc_id:
        return jsonify({"error": "Missing query or doc_id"}), 400

    doc = get_document(doc_id)
    if not doc or doc["workspace_id"] != workspace_id or doc["status"] != "indexed":
        return jsonify({"error": "Document is not yet indexed or not accessible."}), 400

    history = get_conversation(session_id)

    try:
        from core.graph.rag_graph import run_rag_pipeline
        result = run_rag_pipeline(
            query=query,
            doc_id=doc_id,
            conversation_history=history,
            workspace_id=workspace_id
        )

        save_conversation(session_id, workspace_id, result["messages"])

        return jsonify({
            "success": True,
            "session_id": session_id,
            "query": query,
            "rewritten_query": result.get("rewritten_query", query),
            "answer": result["answer"],
            "citations": result["citations"],
            "confidence": result["confidence"],
            "is_grounded": result["is_grounded"],
            "not_found": result.get("not_found", False),
            "debug": result.get("debug", {})
        })
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/chat/stream", methods=["POST"])
@login_required
def chat_stream():
    data = request.get_json()
    query = data.get("query", "").strip()
    doc_id = data.get("doc_id", "").strip()
    session_id = data.get("session_id", str(uuid.uuid4()))
    workspace_id = session["workspace_id"]

    if not query or not doc_id:
        return jsonify({"error": "Missing query or doc_id"}), 400

    doc = get_document(doc_id)
    if not doc or doc["workspace_id"] != workspace_id or doc["status"] != "indexed":
        return jsonify({"error": "Document not yet indexed"}), 400

    history = get_conversation(session_id)

    def generate():
        try:
            yield f"data: {json.dumps({'type': 'status', 'message': 'Rewriting query...'})}\n\n"

            from core.graph.rag_graph import run_rag_pipeline
            result = run_rag_pipeline(
                query=query,
                doc_id=doc_id,
                conversation_history=history,
                workspace_id=workspace_id
            )

            save_conversation(session_id, workspace_id, result["messages"])

            answer = result["answer"]
            words = answer.split(" ")

            yield f"data: {json.dumps({'type': 'status', 'message': 'Generating answer...'})}\n\n"

            accumulated = ""
            for i, word in enumerate(words):
                accumulated += word + " "
                if i % 5 == 0 or i == len(words) - 1:
                    yield f"data: {json.dumps({'type': 'token', 'content': accumulated})}\n\n"

            yield f"data: {json.dumps({'type': 'complete', 'data': result})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

@app.route("/api/clear-conversation", methods=["POST"])
@login_required
def clear_conversation_route():
    data = request.get_json()
    session_id = data.get("session_id", "")
    if session_id:
        clear_conversation(session_id)
    return jsonify({"success": True, "message": "Conversation cleared."})

# ─────────────────────────────────────────────
# DOCUMENT MANAGEMENT
# ─────────────────────────────────────────────

@app.route("/api/documents")
@login_required
def list_documents():
    docs = list_workspace_documents(session["workspace_id"])
    
    # We still need the chunk count from Qdrant? No, we store it in MongoDB.
    formatted = []
    for d in docs:
        formatted.append({
            "doc_id": d["_id"],
            "filename": d.get("file_name", d["_id"]),
            "status": d.get("status", "unknown"),
            "total_pages": d.get("total_pages", 0),
            "vector_count": d.get("total_chunks", 0)
        })
    return jsonify({"documents": formatted})

@app.route("/api/documents/<doc_id>", methods=["DELETE"])
@login_required
def delete_document_route(doc_id):
    doc = get_document(doc_id)
    if not doc or doc["workspace_id"] != session["workspace_id"]:
        return jsonify({"error": "Not found"}), 404

    from core.ingestion.indexer import delete_document_index
    try:
        delete_document_index(doc_id, session["workspace_id"])
        delete_document(doc_id)
        if doc_id in ingestion_status_live:
            del ingestion_status_live[doc_id]
        return jsonify({"success": True, "message": f"Document {doc_id} deleted."})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/page-count/<doc_id>")
@login_required
def get_page_count(doc_id):
    doc = get_document(doc_id)
    if not doc or doc["workspace_id"] != session["workspace_id"]:
        return jsonify({"error": "Not found"}), 404
        
    return jsonify({"page_count": doc.get("total_pages", 0), "doc_id": doc_id})

# ─────────────────────────────────────────────
# PDF VIEWER + HIGHLIGHTING (Keep Unchanged)
# ─────────────────────────────────────────────

def _find_pdf_by_doc_id(doc_id: str) -> str | None:
    upload_dir = UPLOAD_FOLDER
    for fname in os.listdir(upload_dir):
        if fname.lower().endswith(".pdf"):
            if get_doc_id(fname) == doc_id:
                return os.path.join(upload_dir, fname)
    return None

@app.route("/api/pdf/<doc_id>")
@login_required
def serve_pdf(doc_id):
    doc = get_document(doc_id)
    if not doc or doc["workspace_id"] != session["workspace_id"]:
        return jsonify({"error": "Not found"}), 404
        
    pdf_path = _find_pdf_by_doc_id(doc_id)
    if not pdf_path:
        return jsonify({"error": "PDF not found"}), 404
    return send_file(pdf_path, mimetype="application/pdf")

@app.route("/api/page-render/<doc_id>/<int:page_num>")
@login_required
def render_page(doc_id, page_num):
    pdf_path = _find_pdf_by_doc_id(doc_id)
    if not pdf_path:
        return jsonify({"error": "PDF not found"}), 404

    chunk_ids_param = request.args.get("chunks", "")
    chunk_ids = [c.strip() for c in chunk_ids_param.split(",") if c.strip()]

    bboxes = []
    if chunk_ids:
        # We need to query Qdrant to get the bboxes. 
        try:
            from core.ingestion.indexer import get_qdrant_client
            from qdrant_client.models import Filter, FieldCondition, MatchValue
            client = get_qdrant_client()
            results = client.scroll(
                collection_name=session["workspace_id"],
                scroll_filter=Filter(
                    must=[
                        FieldCondition(key="page_number", match=MatchValue(value=page_num)),
                        FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                    ]
                ),
                limit=100,
                with_payload=True
            )
            points, _ = results
            for point in points:
                if point.payload.get("chunk_id") in chunk_ids:
                    bbox = point.payload.get("bbox")
                    if bbox:
                        bboxes.append(bbox)
        except Exception as e:
            logger.error(f"BBox fetch error: {e}")

    try:
        from core.ingestion.pdf_extractor import render_page_with_highlights
        image_bytes = render_page_with_highlights(pdf_path, page_num, bboxes)
        return send_file(
            __import__("io").BytesIO(image_bytes),
            mimetype="image/png",
            download_name=f"page_{page_num}.png"
        )
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/chunk-info/<doc_id>/<chunk_id>")
@login_required
def get_chunk_info(doc_id, chunk_id):
    try:
        from core.ingestion.indexer import get_qdrant_client
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        client = get_qdrant_client()
        results = client.scroll(
            collection_name=session["workspace_id"],
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="chunk_id", match=MatchValue(value=chunk_id)),
                    FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
                ]
            ),
            limit=1,
            with_payload=True
        )
        points, _ = results
        if not points:
            return jsonify({"error": "Chunk not found"}), 404
        payload = points[0].payload
        return jsonify({
            "chunk_id": payload.get("chunk_id"),
            "page_number": payload.get("page_number"),
            "text": payload.get("text"),
            "bbox": payload.get("bbox"),
            "page_width": payload.get("page_width"),
            "page_height": payload.get("page_height"),
            "content_type": payload.get("content_type"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ─────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500

# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"
    logger.info(f"Starting RAG Document Intelligence on port {port}")
    app.run(host="0.0.0.0", port=port, debug=debug, threaded=True)