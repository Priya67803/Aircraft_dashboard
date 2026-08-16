"""
app.py
======
Flask backend for the PHNet-CMAPSS Mission Control dashboard.

Endpoints
---------
GET  /                 -> dashboard UI
POST /api/run          -> starts a pipeline run in a background thread
                           body: {"mode": "quick"|"full", "seed": int}
GET  /api/status       -> {"running", "done", "error", "stage_index",
                            "stage_name", "total_stages", "logs" (new
                            lines since `since`), "results"}
                           query param `since` = number of log lines
                           already seen by the client
POST /api/stop         -> best-effort cancel flag (checked between
                           epochs is out of scope for a demo trainer,
                           so this just marks the run as user-stopped
                           for the UI; the thread finishes its current
                           stage)
"""
import threading
import time
import uuid

from flask import Flask, jsonify, render_template, request

from pipeline.runner import run_pipeline_safe

app = Flask(__name__)

STATE_LOCK = threading.Lock()
STATE = {
    "run_id": None,
    "running": False,
    "done": False,
    "error": None,
    "stage_index": -1,
    "stage_name": "",
    "total_stages": 11,
    "logs": [],
    "results": None,
    "mode": "quick",
    "started_at": None,
}


def _reset_state(mode):
    STATE.update({
        "run_id": str(uuid.uuid4())[:8],
        "running": True,
        "done": False,
        "error": None,
        "stage_index": -1,
        "stage_name": "Initializing…",
        "logs": [],
        "results": None,
        "mode": mode,
        "started_at": time.time(),
    })


def _progress_cb(event):
    with STATE_LOCK:
        if event["type"] == "stage":
            STATE["stage_index"] = event["index"]
            STATE["stage_name"] = event["name"]
            STATE["total_stages"] = event["total"]
        elif event["type"] == "log":
            STATE["logs"].append(event["text"])
            if len(STATE["logs"]) > 4000:
                STATE["logs"] = STATE["logs"][-4000:]


def _worker(mode, seed):
    outcome = run_pipeline_safe(mode=mode, progress_cb=_progress_cb, seed=seed)
    with STATE_LOCK:
        STATE["running"] = False
        STATE["done"] = True
        if outcome["ok"]:
            STATE["results"] = outcome["results"]
        else:
            STATE["error"] = outcome["error"]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/run", methods=["POST"])
def api_run():
    with STATE_LOCK:
        if STATE["running"]:
            return jsonify({"ok": False, "message": "A run is already in progress."}), 409
        body = request.get_json(silent=True) or {}
        mode = body.get("mode", "quick")
        seed = body.get("seed", 42)
        if mode not in ("quick", "full"):
            mode = "quick"
        _reset_state(mode)
    t = threading.Thread(target=_worker, args=(mode, seed), daemon=True)
    t.start()
    return jsonify({"ok": True, "run_id": STATE["run_id"]})


@app.route("/api/status")
def api_status():
    since = request.args.get("since", default=0, type=int)
    with STATE_LOCK:
        new_logs = STATE["logs"][since:]
        payload = {
            "run_id": STATE["run_id"],
            "running": STATE["running"],
            "done": STATE["done"],
            "error": STATE["error"],
            "stage_index": STATE["stage_index"],
            "stage_name": STATE["stage_name"],
            "total_stages": STATE["total_stages"],
            "mode": STATE["mode"],
            "log_count": len(STATE["logs"]),
            "new_logs": new_logs,
            "results": STATE["results"] if STATE["done"] else None,
        }
    return jsonify(payload)


@app.route("/healthz")
def healthz():
    return jsonify({"ok": True})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5050, debug=False, threaded=True)
