import http.server
import json
import os
import socketserver
import subprocess
import urllib.parse
import webbrowser
from typing import Dict

from .session import TuneSession


class VistuneHandler(http.server.SimpleHTTPRequestHandler):
    session: TuneSession
    tensorboards: Dict[str, tuple] = {}

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            self.send_html(self.page())
            return
        if parsed.path == "/api/session":
            self.send_json(self.load_session())
            return
        if parsed.path == "/api/tensorboard":
            query = urllib.parse.parse_qs(parsed.query)
            logdir = query.get("logdir", [None])[0]
            self.send_json({"url": self.open_tensorboard(logdir)})
            return
        self.send_error(404)

    def log_message(self, format, *args):
        return

    def send_json(self, data):
        payload = json.dumps(data, indent=2).encode("utf8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_html(self, html: str):
        payload = html.encode("utf8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def load_session(self):
        manifest = self.session.read_json("manifest.json")
        state = self.session.read_json("state.json")
        groups = []
        group_dir = self.session.path("groups")
        if os.path.exists(group_dir):
            for name in sorted(os.listdir(group_dir)):
                if name.endswith(".json"):
                    with open(os.path.join(group_dir, name), "r", encoding="utf8") as fh:
                        groups.append(json.load(fh))
        return {"manifest": manifest, "state": state, "groups": groups}

    def open_tensorboard(self, logdir):
        if not logdir:
            logdir = os.path.join("logs", self.session.read_json("manifest.json")["description"])
        logdir = os.path.abspath(logdir)
        if logdir not in self.tensorboards or self.tensorboards[logdir][0].poll() is not None:
            port = 6006 + len(self.tensorboards)
            process = subprocess.Popen(
                ["tensorboard", "--logdir", logdir, "--port", str(port)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            url = f"http://localhost:{port}"
            self.tensorboards[logdir] = (process, url)
            return url
        return self.tensorboards[logdir][1]

    @staticmethod
    def page():
        return r"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>freerec vistune</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f7f7f4; color: #242424; }
    main { max-width: 1180px; margin: 0 auto; padding: 32px 20px 56px; }
    h1 { font-size: 28px; margin: 0 0 8px; }
    h2 { font-size: 22px; margin-top: 36px; border-top: 1px solid #ddd; padding-top: 24px; }
    .meta, .analysis { background: white; border: 1px solid #deded8; border-radius: 8px; padding: 16px; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 16px 0; }
    .stat { background: white; border: 1px solid #deded8; border-radius: 8px; padding: 12px; }
    .label { color: #666; font-size: 12px; text-transform: uppercase; }
    .value { font-weight: 650; margin-top: 4px; overflow-wrap: anywhere; }
    table { width: 100%; border-collapse: collapse; background: white; border: 1px solid #deded8; border-radius: 8px; overflow: hidden; }
    th, td { padding: 8px 10px; border-bottom: 1px solid #eee; text-align: left; vertical-align: top; font-size: 13px; }
    th { background: #eeeeea; }
    pre { white-space: pre-wrap; word-break: break-word; margin: 0; }
    button { border: 1px solid #1f5f8b; background: #276f9f; color: white; border-radius: 6px; padding: 8px 12px; cursor: pointer; }
    canvas { width: 100%; height: 240px; background: white; border: 1px solid #deded8; border-radius: 8px; }
  </style>
</head>
<body>
<main>
  <h1>freerec vistune</h1>
  <section id="app">Loading...</section>
</main>
<script>
function esc(v) { return String(v ?? '').replace(/[&<>]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
function draw(canvas, trials) {
  const ctx = canvas.getContext('2d'), w = canvas.width = canvas.clientWidth * devicePixelRatio, h = canvas.height = 240 * devicePixelRatio;
  ctx.scale(devicePixelRatio, devicePixelRatio); ctx.clearRect(0,0,w,h);
  const vals = trials.map(t => t.best_value ?? null).filter(v => typeof v === 'number');
  if (!vals.length) { ctx.fillText('No metric data yet', 20, 32); return; }
  const min = Math.min(...vals), max = Math.max(...vals), pad = 28, width = canvas.clientWidth - pad * 2, height = 240 - pad * 2;
  ctx.strokeStyle = '#d0d0ca'; ctx.beginPath(); ctx.moveTo(pad, pad); ctx.lineTo(pad, pad + height); ctx.lineTo(pad + width, pad + height); ctx.stroke();
  ctx.strokeStyle = '#276f9f'; ctx.lineWidth = 2; ctx.beginPath();
  trials.forEach((t, i) => {
    const v = t.best_value; if (typeof v !== 'number') return;
    const x = pad + (trials.length === 1 ? 0 : i * width / (trials.length - 1));
    const y = pad + height - ((v - min) / ((max - min) || 1)) * height;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  ctx.stroke();
}
function metricValue(trial, metric) {
  for (const mode of ['valid', 'best', 'test']) if (trial.metrics?.[mode]?.[metric] != null) return trial.metrics[mode][metric];
  return '';
}
fetch('/api/session').then(r => r.json()).then(data => {
  const m = data.manifest, s = data.state;
  document.getElementById('app').innerHTML = `
    <div class="meta"><div class="grid">
      <div class="stat"><div class="label">Description</div><div class="value">${esc(m.description)}</div></div>
      <div class="stat"><div class="label">Dataset</div><div class="value">${esc(m.dataset)}</div></div>
      <div class="stat"><div class="label">Status</div><div class="value">${esc(s.status || m.status)}</div></div>
      <div class="stat"><div class="label">Current Group</div><div class="value">${esc(s.current_group)}</div></div>
      <div class="stat"><div class="label">which4best</div><div class="value">${esc(m.which4best)}</div></div>
    </div></div>
    ${data.groups.map(g => {
      const trials = g.trials || [];
      return `<h2>${esc(g.name)}</h2>
      <div class="grid">
        <div class="stat"><div class="label">Best Trial</div><div class="value">${esc(g.best_trial_id)}</div></div>
        <div class="stat"><div class="label">Best Value</div><div class="value">${esc(g.best_value)}</div></div>
        <div class="stat"><div class="label">Best Params</div><div class="value"><pre>${esc(JSON.stringify(g.best_params, null, 2))}</pre></div></div>
      </div>
      <canvas id="chart-${esc(g.name)}"></canvas>
      <p><button onclick="fetch('/api/tensorboard').then(r=>r.json()).then(d=>window.open(d.url,'_blank'))">Open TensorBoard</button></p>
      <table><thead><tr><th>ID</th><th>Status</th><th>Params</th><th>Metrics</th><th>Log Path</th></tr></thead>
      <tbody>${trials.map(t => `<tr><td>${esc(t.id)}</td><td>${esc(t.status)}</td><td><pre>${esc(JSON.stringify(t.params, null, 2))}</pre></td><td><pre>${esc(JSON.stringify(t.metrics, null, 2))}</pre></td><td>${esc(t.log_path)}</td></tr>`).join('')}</tbody></table>
      <h3>LLM Analysis</h3><div class="analysis"><pre>${esc(JSON.stringify(g.llm_analysis || {}, null, 2))}</pre></div>`;
    }).join('')}`;
  data.groups.forEach(g => draw(document.getElementById(`chart-${g.name}`), (g.trials || []).map(t => ({...t, best_value: metricValue(t, m.which4best)}))));
});
</script>
</body>
</html>"""


def serve_vistune(description: str, session_id=None, port: int = 8765, open_browser: bool = True):
    session = TuneSession.load(description, session_id)
    handler = type("Handler", (VistuneHandler,), {"session": session})
    with socketserver.TCPServer(("", port), handler) as httpd:
        url = f"http://localhost:{port}"
        if open_browser:
            webbrowser.open(url)
        print(f"Serving vistune at {url}")
        httpd.serve_forever()
