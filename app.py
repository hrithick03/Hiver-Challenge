"""
Interactive Web Application for Apple Support AI Agent.
Launches a sleek, Apple-inspired Web UI to test arbitrary customer queries.
Run with: python app.py
"""

import sys
import os
import webbrowser
from pathlib import Path

# Ensure root directory is on sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from src.agent import AppleSupportAgent

app = FastAPI(title="Apple Support AI Agent Console")
agent = AppleSupportAgent()

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Apple Support AI Agent — Interactive Console</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link href="https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@400;500;600;700&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0f1117;
      --card-bg: rgba(26, 29, 39, 0.85);
      --card-border: rgba(255, 255, 255, 0.08);
      --primary: #0071e3;
      --primary-hover: #0077ed;
      --text: #f5f5f7;
      --text-muted: #86868b;
      --success: #30d158;
      --danger: #ff453a;
      --warning: #ffd60a;
      --purple: #bf5af2;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 32px 16px;
      background-image: radial-gradient(circle at 50% 0%, rgba(0, 113, 227, 0.15), transparent 45%);
    }
    .container {
      width: 100%;
      max-width: 900px;
    }
    header {
      text-align: center;
      margin-bottom: 32px;
    }
    .logo-badge {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      padding: 6px 16px;
      border-radius: 999px;
      background: rgba(0, 113, 227, 0.12);
      border: 1px solid rgba(0, 113, 227, 0.3);
      color: #2997ff;
      font-size: 13px;
      font-weight: 600;
      margin-bottom: 12px;
    }
    h1 {
      font-size: 32px;
      font-weight: 700;
      letter-spacing: -0.5px;
      margin-bottom: 8px;
    }
    p.subtitle {
      color: var(--text-muted);
      font-size: 15px;
      max-width: 600px;
      margin: 0 auto;
    }
    .card {
      background: var(--card-bg);
      backdrop-filter: blur(20px);
      border: 1px solid var(--card-border);
      border-radius: 18px;
      padding: 24px;
      box-shadow: 0 12px 36px rgba(0, 0, 0, 0.4);
      margin-bottom: 24px;
    }
    .input-group {
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    label {
      font-size: 13px;
      font-weight: 600;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: var(--text-muted);
    }
    textarea {
      width: 100%;
      background: rgba(15, 17, 23, 0.7);
      border: 1px solid var(--card-border);
      border-radius: 12px;
      padding: 14px 16px;
      color: var(--text);
      font-family: inherit;
      font-size: 15px;
      resize: vertical;
      min-height: 90px;
      outline: none;
      transition: border-color 0.2s, box-shadow 0.2s;
    }
    textarea:focus {
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(0, 113, 227, 0.25);
    }
    .btn-row {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-top: 8px;
      flex-wrap: wrap;
      gap: 12px;
    }
    .btn-primary {
      background: var(--primary);
      color: white;
      border: none;
      border-radius: 10px;
      padding: 12px 24px;
      font-size: 15px;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 8px;
      transition: background 0.2s, transform 0.1s;
    }
    .btn-primary:hover { background: var(--primary-hover); }
    .btn-primary:active { transform: scale(0.98); }
    .btn-primary:disabled { opacity: 0.5; cursor: not-allowed; }
    
    .presets-title {
      font-size: 12px;
      font-weight: 600;
      color: var(--text-muted);
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .presets-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }
    .preset-btn {
      background: rgba(255, 255, 255, 0.05);
      border: 1px solid var(--card-border);
      color: var(--text);
      border-radius: 8px;
      padding: 6px 12px;
      font-size: 12px;
      cursor: pointer;
      transition: background 0.2s, border-color 0.2s;
    }
    .preset-btn:hover {
      background: rgba(255, 255, 255, 0.1);
      border-color: rgba(255, 255, 255, 0.2);
    }

    /* Result Section */
    #results { display: none; }
    .status-row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 16px;
      margin-bottom: 20px;
    }
    @media (max-width: 650px) {
      .status-row { grid-template-columns: 1fr; }
    }
    .metric-card {
      background: rgba(15, 17, 23, 0.6);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 16px;
    }
    .metric-label {
      font-size: 11px;
      text-transform: uppercase;
      color: var(--text-muted);
      font-weight: 600;
      margin-bottom: 6px;
    }
    .metric-value {
      font-size: 18px;
      font-weight: 700;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 12px;
      font-weight: 600;
    }
    .badge-auto { background: rgba(48, 209, 88, 0.15); color: var(--success); border: 1px solid rgba(48, 209, 88, 0.3); }
    .badge-escalate { background: rgba(255, 69, 58, 0.15); color: var(--danger); border: 1px solid rgba(255, 69, 58, 0.3); }
    .badge-intent { background: rgba(0, 113, 227, 0.15); color: #2997ff; border: 1px solid rgba(0, 113, 227, 0.3); }

    .reply-box {
      background: linear-gradient(145deg, rgba(0, 113, 227, 0.08), rgba(255, 255, 255, 0.02));
      border: 1px solid rgba(0, 113, 227, 0.25);
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 20px;
    }
    .reply-title {
      font-size: 12px;
      text-transform: uppercase;
      color: #2997ff;
      font-weight: 700;
      letter-spacing: 0.5px;
      margin-bottom: 10px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }
    .reply-text {
      font-size: 16px;
      line-height: 1.5;
      font-weight: 500;
      color: #ffffff;
    }
    .copy-btn {
      background: rgba(255, 255, 255, 0.08);
      border: 1px solid var(--card-border);
      color: var(--text-muted);
      border-radius: 6px;
      padding: 4px 8px;
      font-size: 11px;
      cursor: pointer;
    }
    .copy-btn:hover { color: var(--text); background: rgba(255, 255, 255, 0.15); }

    .retrieval-section {
      margin-top: 16px;
    }
    .retrieval-item {
      background: rgba(15, 17, 23, 0.5);
      border-left: 3px solid var(--primary);
      padding: 12px 14px;
      border-radius: 0 10px 10px 0;
      margin-bottom: 8px;
      font-size: 13px;
    }
    .retrieval-meta {
      display: flex;
      justify-content: space-between;
      color: var(--text-muted);
      font-size: 11px;
      margin-bottom: 4px;
    }
    .loading-spinner {
      display: none;
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255, 255, 255, 0.3);
      border-top-color: white;
      border-radius: 50%;
      animation: spin 0.6s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <div class="logo-badge"> Apple Support AI Agent</div>
      <h1>Interactive Testing Console</h1>
      <p class="subtitle">Test arbitrary customer tweets with real-time Intent Classification, Explainable Triage Escalation, and Grounded Reply Drafting.</p>
    </header>

    <div class="card">
      <div class="input-group">
        <label for="tweetInput">Incoming Customer Tweet</label>
        <textarea id="tweetInput" placeholder="Type or paste any customer support tweet here..."></textarea>
      </div>

      <div class="btn-row">
        <button id="sendBtn" class="btn-primary" onclick="processTweet()">
          <span class="loading-spinner" id="spinner"></span>
          <span>Run Agent Pipeline</span>
        </button>
        <span id="latencyTag" style="font-size: 12px; color: var(--text-muted);"></span>
      </div>

      <div style="margin-top: 20px;">
        <div class="presets-title">Quick Test Presets:</div>
        <div class="presets-grid" id="presetsGrid"></div>
      </div>
    </div>

    <div id="results" class="card">
      <div class="status-row">
        <div class="metric-card">
          <div class="metric-label">1. Classified Intent</div>
          <div class="metric-value">
            <span id="intentBadge" class="badge badge-intent"></span>
            <span id="intentConf" style="font-size: 13px; color: var(--text-muted); font-weight: 500;"></span>
          </div>
          <div id="intentReason" style="font-size: 12px; color: var(--text-muted); margin-top: 8px;"></div>
        </div>

        <div class="metric-card">
          <div class="metric-label">2. Triage & Escalation Decision</div>
          <div class="metric-value">
            <span id="triageBadge" class="badge"></span>
          </div>
          <div id="triageReason" style="font-size: 12px; color: var(--text-muted); margin-top: 8px;"></div>
        </div>
      </div>

      <div class="reply-box">
        <div class="reply-title">
          <span>3. Official Grounded Draft Reply</span>
          <button class="copy-btn" onclick="copyReply()">Copy Reply</button>
        </div>
        <div class="reply-text" id="draftReply"></div>
      </div>

      <div class="retrieval-section">
        <div class="presets-title">Retrieved Historical Apple Resolutions (Grounding Context)</div>
        <div id="retrievalList"></div>
      </div>
    </div>
  </div>

  <script>
    const PRESETS = [
      {
        title: "⚡ Battery Drain Post-Update",
        text: "Ever since updating to iOS 11.1, my battery drains from 100% to 20% in two hours."
      },
      {
        title: "🔨 Cracked Screen / Hardware Damage",
        text: "Dropped my iPhone X on concrete and the front screen is completely shattered with green flickering lines."
      },
      {
        title: "💳 Unauthorized Charge / Billing Dispute",
        text: "Apple charged my credit card $49.99 for a subscription I cancelled last week! I demand an immediate refund."
      },
      {
        title: "🔒 Apple ID Locked / Security",
        text: "My Apple ID has been locked for security reasons and iforgot is not sending the reset code."
      },
      {
        title: "🚨 PII Trap / Safety Hazard",
        text: "My email is alex92@gmail.com and serial is F17TM384HFL4. Unlock my Activation Lock now!"
      },
      {
        title: "🎭 Sarcastic Rant",
        text: "Thanks Apple for the amazing update! My phone is now an expensive paperweight and stove. 10/10 guys."
      }
    ];

    // Populate presets
    const grid = document.getElementById("presetsGrid");
    PRESETS.forEach(p => {
      const btn = document.createElement("button");
      btn.className = "preset-btn";
      btn.textContent = p.title;
      btn.onclick = () => {
        document.getElementById("tweetInput").value = p.text;
        processTweet();
      };
      grid.appendChild(btn);
    });

    async function processTweet() {
      const input = document.getElementById("tweetInput").value.trim();
      if (!input) return;

      const btn = document.getElementById("sendBtn");
      const spinner = document.getElementById("spinner");
      btn.disabled = true;
      spinner.style.display = "inline-block";

      try {
        const res = await fetch("/api/process", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ tweet: input })
        });
        const data = await res.json();
        renderResults(data);
      } catch (err) {
        alert("Error executing pipeline: " + err.message);
      } finally {
        btn.disabled = false;
        spinner.style.display = "none";
      }
    }

    function renderResults(data) {
      document.getElementById("results").style.display = "block";
      document.getElementById("intentBadge").textContent = data.intent;
      document.getElementById("intentConf").textContent = `(Confidence: ${(data.intent_confidence * 100).toFixed(0)}%)`;
      document.getElementById("intentReason").textContent = data.intent_reasoning;

      const triageBadge = document.getElementById("triageBadge");
      triageBadge.textContent = data.triage_decision;
      triageBadge.className = data.escalate ? "badge badge-escalate" : "badge badge-auto";
      document.getElementById("triageReason").textContent = "Reason: " + data.escalation_reason;

      document.getElementById("draftReply").textContent = data.draft_reply;
      document.getElementById("latencyTag").textContent = `Latency: ${data.latency_ms} ms`;

      const retList = document.getElementById("retrievalList");
      retList.innerHTML = "";
      (data.retrieved_context || []).slice(0, 2).forEach((hit, idx) => {
        const item = document.createElement("div");
        item.className = "retrieval-item";
        item.innerHTML = `
          <div class="retrieval-meta">
            <span>Historical Pair #${idx + 1}</span>
            <span>Relevance Score: ${hit.score.toFixed(3)}</span>
          </div>
          <div style="color: #a1a1a6; margin-bottom: 4px;"><strong>Customer:</strong> "${hit.customer_query}"</div>
          <div><strong>Apple Resolution:</strong> "${hit.historical_resolution}"</div>
        `;
        retList.appendChild(item);
      });

      document.getElementById("results").scrollIntoView({ behavior: 'smooth' });
    }

    function copyReply() {
      const text = document.getElementById("draftReply").textContent;
      navigator.clipboard.writeText(text).then(() => {
        alert("Draft reply copied to clipboard!");
      });
    }
  </script>
</body>
</html>
"""


class TweetRequest(BaseModel):
    tweet: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    return HTMLResponse(content=HTML_CONTENT)


@app.post("/api/process")
async def process_tweet_endpoint(req: TweetRequest):
    result = agent.process_message(req.tweet)
    return JSONResponse(content=result)


def start_server():
    port = 8000
    print("\n" + "=" * 60)
    print(f"   Apple Support AI Agent — Interactive Web UI")
    print(f"  Access in browser at: http://localhost:{port}")
    print("=" * 60 + "\n")
    webbrowser.open(f"http://localhost:{port}")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


if __name__ == "__main__":
    start_server()
