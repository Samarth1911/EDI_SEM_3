"""Optional: serve the dashboard on http://localhost:8000 (stdlib only). Run:  python frontend/serve.py
The dashboard is fully client-side, so this is a convenience, not a requirement."""
import http.server, socketserver, webbrowser, os, threading
os.chdir(os.path.dirname(os.path.abspath(__file__)))
PORT = int(os.environ.get("PORT", "5500"))   # 8000 is reserved for the FastAPI backend (uvicorn)
class Q(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass
socketserver.TCPServer.allow_reuse_address = True
with socketserver.TCPServer(("127.0.0.1", PORT), Q) as srv:
    print(f"Milestone demo: http://localhost:{PORT}/   Live dashboard: http://localhost:{PORT}/dashboard.html   (Ctrl+C to stop)")
    threading.Timer(0.6, lambda: webbrowser.open(f"http://localhost:{PORT}")).start()
    try: srv.serve_forever()
    except KeyboardInterrupt: pass