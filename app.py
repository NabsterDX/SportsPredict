from flask import Flask, render_template_string, request, jsonify
import numpy as np
import pandas as pd
import requests
from io import StringIO
from collections import defaultdict
import math
from datetime import date, timedelta
import threading
import pickle
import os
import difflib
import re

app = Flask(__name__)
MODEL_FILE = "soccer_model.pkl"

MODEL = {
    "ready": False,
    "attack": {},
    "defense": {},
    "home_adv": 1.28,
    "message": "Starting...",
    "today_preds": [],
    "all_teams": [],
    "progress": "Initializing"
}

HALF_LIFE_DAYS = 45
LEAGUES = ['E0','E1','E2','E3','SP1','SP2','I1','I2','D1','D2','F1','F2','N1','P1','T1','B1','SC0','G1']
SEASONS = ['2425','2526','2627']

def poisson_pmf(k, lam):
    return np.exp(-lam) * (lam ** k) / math.factorial(k)

def time_weight(match_date, today):
    if pd.isna(match_date):
        return 0.3
    days_ago = max(0, (today - match_date).days)
    return math.exp(-math.log(2) * days_ago / HALF_LIFE_DAYS)

def save_model():
    data = {
        "attack": MODEL["attack"],
        "defense": MODEL["defense"],
        "home_adv": MODEL["home_adv"],
        "version": "paste_v3"
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(data, f)

def load_saved_model():
    if os.path.exists(MODEL_FILE):
        try:
            with open(MODEL_FILE, "rb") as f:
                data = pickle.load(f)
            if data.get("version") != "paste_v3":
                return False
            MODEL["attack"] = data["attack"]
            MODEL["defense"] = data["defense"]
            MODEL["home_adv"] = data["home_adv"]
            MODEL["all_teams"] = sorted(MODEL["attack"].keys())
            MODEL["ready"] = True
            return True
        except:
            return False
    return False

def find_team(name):
    if not name or not MODEL["all_teams"]:
        return None
    name = name.strip()
    if name in MODEL["attack"]:
        return name
    lower_map = {t.lower(): t for t in MODEL["all_teams"]}
    if name.lower() in lower_map:
        return lower_map[name.lower()]
    matches = difflib.get_close_matches(name, MODEL["all_teams"], n=1, cutoff=0.65)
    if matches:
        return matches[0]
    for t in MODEL["all_teams"]:
        if name.lower() in t.lower() or t.lower() in name.lower():
            return t
    return None

def train_model():
    global MODEL
    try:
        MODEL["progress"] = "Downloading historical data..."
        MODEL["message"] = "Training model • Downloading data..."
        frames = []
        total = len(LEAGUES) * len(SEASONS)
        done = 0
        for season in SEASONS:
            for league in LEAGUES:
                url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"
                try:
                    r = requests.get(url, timeout=10)
                    if r.status_code == 200:
                        content = r.content.decode('utf-8-sig', errors='replace')
                        df = pd.read_csv(StringIO(content))
                        df = df.rename(columns={
                            'Date':'date','HomeTeam':'home_team','AwayTeam':'away_team',
                            'FTHG':'home_goals','FTAG':'away_goals'
                        })
                        if not all(c in df.columns for c in ['date','home_team','away_team','home_goals','away_goals']):
                            done += 1
                            continue
                        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
                        df = df.dropna(subset=['date','home_team','away_team','home_goals','away_goals'])
                        df['home_goals'] = df['home_goals'].astype(int)
                        df['away_goals'] = df['away_goals'].astype(int)
                        frames.append(df[['date','home_team','away_team','home_goals','away_goals']])
                except:
                    pass
                done += 1
                pct = int(done / total * 100)
                MODEL["progress"] = f"Downloading data... {done}/{total} ({pct}%)"
                MODEL["message"] = f"Training • {MODEL['progress']}"

        if not frames:
            MODEL["message"] = "Failed to load data"
            MODEL["progress"] = "Failed"
            return

        MODEL["progress"] = "Calculating team strengths..."
        MODEL["message"] = "Training • Calculating strengths (this takes 1-3 min)..."
        data = pd.concat(frames, ignore_index=True).sort_values('date')
        today = pd.Timestamp(date.today())
        data['weight'] = data['date'].apply(lambda d: time_weight(d, today))
        teams = sorted(set(data['home_team']) | set(data['away_team']))
        attack = {t: 1.0 for t in teams}
        defense = {t: 1.0 for t in teams}
        home_adv = 1.28

        for it in range(16):
            MODEL["progress"] = f"Calculating strengths... step {it+1}/16"
            MODEL["message"] = f"Training • {MODEL['progress']}"
            new_attack = defaultdict(float)
            weight_sum = defaultdict(float)
            for _, row in data.iterrows():
                w = row['weight']
                new_attack[row['home_team']] += w * row['home_goals'] / (defense[row['away_team']] * home_adv + 1e-6)
                new_attack[row['away_team']] += w * row['away_goals'] / (defense[row['home_team']] + 1e-6)
                weight_sum[row['home_team']] += w
                weight_sum[row['away_team']] += w
            for t in teams:
                attack[t] = new_attack[t] / weight_sum[t] if weight_sum[t] > 0 else 1.0

            new_defense = defaultdict(float)
            weight_sum = defaultdict(float)
            for _, row in data.iterrows():
                w = row['weight']
                new_defense[row['away_team']] += w * row['home_goals'] / (attack[row['home_team']] * home_adv + 1e-6)
                new_defense[row['home_team']] += w * row['away_goals'] / (attack[row['away_team']] + 1e-6)
                weight_sum[row['home_team']] += w
                weight_sum[row['away_team']] += w
            for t in teams:
                defense[t] = new_defense[t] / weight_sum[t] if weight_sum[t] > 0 else 1.0

            avg_att = np.mean(list(attack.values())) or 1.0
            avg_def = np.mean(list(defense.values())) or 1.0
            for t in teams:
                attack[t] /= avg_att
                defense[t] /= avg_def

        MODEL["attack"] = attack
        MODEL["defense"] = defense
        MODEL["home_adv"] = home_adv
        MODEL["all_teams"] = teams
        MODEL["ready"] = True
        save_model()
        MODEL["progress"] = "Loading upcoming fixtures..."
        MODEL["message"] = "Almost done • Loading fixtures..."
        load_todays_predictions()
        print(f"Model ready • {len(teams)} teams")
    except Exception as e:
        MODEL["message"] = f"Error: {e}"
        MODEL["progress"] = "Error"

def add_leans(p):
    hw, d, aw = p["home_win"], p["draw"], p["away_win"]
    if hw >= d and hw >= aw:
        p["lean"] = "HOME"
        p["lean_pct"] = hw
        p["lean_class"] = "lean-home"
    elif aw >= d and aw >= hw:
        p["lean"] = "AWAY"
        p["lean_pct"] = aw
        p["lean_class"] = "lean-away"
    else:
        p["lean"] = "DRAW"
        p["lean_pct"] = d
        p["lean_class"] = "lean-draw"
    p["strong_home"] = hw >= 55
    p["strong_away"] = aw >= 55
    p["strong_under"] = p["under25"] >= 58
    p["strong_over"] = p["over25"] >= 58
    p["strong_btts_no"] = p["btts_no"] >= 58
    p["strong_btts_yes"] = p["btts_yes"] >= 58
    return p

def predict(home, away):
    if not MODEL["ready"]:
        return None
    h = find_team(home)
    a = find_team(away)
    if not h or not a:
        return None
    lam_h = MODEL["attack"][h] * MODEL["defense"][a] * MODEL["home_adv"]
    lam_a = MODEL["attack"][a] * MODEL["defense"][h]
    matrix = np.zeros((7,7))
    for i in range(7):
        for j in range(7):
            matrix[i,j] = poisson_pmf(i, lam_h) * poisson_pmf(j, lam_a)
    matrix /= matrix.sum()
    home_win = np.tril(matrix, -1).sum()
    draw = np.trace(matrix)
    away_win = np.triu(matrix, 1).sum()
    over25 = sum(matrix[i,j] for i in range(7) for j in range(7) if i+j > 2.5)
    under25 = 1 - over25
    btts_yes = sum(matrix[i,j] for i in range(1,7) for j in range(1,7))
    btts_no = 1 - btts_yes
    flat = [(i, j, matrix[i,j]) for i in range(7) for j in range(7)]
    top = sorted(flat, key=lambda x: x[2], reverse=True)[:4]
    p = {
        "home": h, "away": a,
        "lam_h": round(lam_h, 2), "lam_a": round(lam_a, 2),
        "home_win": round(home_win*100, 1),
        "draw": round(draw*100, 1),
        "away_win": round(away_win*100, 1),
        "over25": round(over25*100, 1),
        "under25": round(under25*100, 1),
        "btts_yes": round(btts_yes*100, 1),
        "btts_no": round(btts_no*100, 1),
        "top": top,
        "strength": max(home_win, draw, away_win)
    }
    return add_leans(p)

def extract_matches_from_text(text):
    """Extract matches + try to capture league/competition"""
    results = []
    text = text.replace('\r', '\n')
    lines = [l.strip() for l in text.split('\n') if l.strip()]

    vs_pattern = re.compile(r'(.+?)\s+[Vv][sS]\.?\s+(.+)', re.IGNORECASE)
    current_league = ""

    for i, line in enumerate(lines):
        # Detect league/competition lines
        lower = line.lower()
        if any(k in lower for k in ['international', 'afcon', 'uefa', 'nations', 'friendly', 'qualification',
                                     'youth', 'league', 'division', 'championship', 'cup', 'trophy',
                                     'premier', 'nacional', 'serie', 'bundesliga', 'ligue']):
            # Clean it a bit
            current_league = re.sub(r'\s+', ' ', line)
            if len(current_league) > 60:
                current_league = current_league[:57] + "..."
            continue

        m = vs_pattern.search(line)
        if m:
            home = m.group(1).strip()
            away = m.group(2).strip()
            # Remove common junk
            for junk in ['more', '•', '|']:
                home = home.replace(junk, '').strip()
                away = away.replace(junk, '').strip()
            if len(home) > 2 and len(away) > 2:
                results.append({
                    "home": home,
                    "away": away,
                    "league": current_league or "Unknown"
                })
    # Deduplicate
    seen = set()
    unique = []
    for item in results:
        key = (item["home"].lower(), item["away"].lower())
        if key not in seen:
            seen.add(key)
            unique.append(item)
    return unique

def load_todays_predictions():
    try:
        r = requests.get("https://www.football-data.co.uk/fixtures.csv", timeout=15)
        content = r.content.decode('utf-8-sig', errors='replace')
        fix = pd.read_csv(StringIO(content))
        fix = fix.rename(columns={
            'Date':'date','HomeTeam':'home_team','AwayTeam':'away_team',
            'Div':'league','Time':'time'
        })
        fix['date'] = pd.to_datetime(fix['date'], dayfirst=True, errors='coerce')
        today = date.today()
        end = today + timedelta(days=3)
        mask = (fix['date'].dt.date >= today) & (fix['date'].dt.date <= end)
        upcoming = fix[mask].dropna(subset=['home_team','away_team'])

        preds = []
        for _, row in upcoming.iterrows():
            p = predict(str(row['home_team']).strip(), str(row['away_team']).strip())
            if p:
                p['league'] = str(row.get('league', ''))
                p['time'] = str(row.get('time', ''))
                p['match_date'] = row['date'].strftime('%a %d %b')
                preds.append(p)
        preds.sort(key=lambda x: x['strength'], reverse=True)
        MODEL["today_preds"] = preds
        MODEL["message"] = f"Ready • {len(MODEL['attack'])} teams • {len(preds)} matches (today+3d)"
        MODEL["progress"] = "Done"
    except Exception as e:
        MODEL["message"] = "Model ready • fixtures error"
        MODEL["progress"] = "Done"

def startup():
    if load_saved_model():
        print("Loaded model")
        MODEL["message"] = "Loaded model • fetching fixtures..."
        MODEL["progress"] = "Loading fixtures..."
        load_todays_predictions()
    else:
        train_model()

threading.Thread(target=startup, daemon=True).start()

HTML = """
<!DOCTYPE html>
<html>
<head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Sports Predict</title>
<style>
:root {
  --bg: #0b1120; --card: #111827; --border: #1f2937;
  --text: #e5e7eb; --muted: #9ca3af; --accent: #38bdf8;
  --green: #4ade80;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: var(--bg); color: var(--text); margin: 0; padding: 0; line-height: 1.4;
}
.header { padding: 14px 14px 0; }
.header h1 { font-size: 1.25rem; margin: 0 0 4px; color: var(--accent); }
.status { font-size: 0.82rem; color: var(--muted); margin-bottom: 12px; }
.tabs {
  display: flex; overflow-x: auto; padding: 0 8px;
  border-bottom: 1px solid var(--border); background: #0f172a;
}
.tab {
  flex: 0 0 auto; padding: 12px 16px; font-size: 0.9rem; font-weight: 600;
  color: var(--muted); background: none; border: none; border-bottom: 3px solid transparent;
  cursor: pointer;
}
.tab.active { color: var(--accent); border-bottom-color: var(--accent); }
.tab-content { display: none; padding: 14px; }
.tab-content.active { display: block; }

.loading-box {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 28px 20px; text-align: center; margin-bottom: 16px;
}
.spinner {
  width: 32px; height: 32px; border: 3px solid #334155;
  border-top-color: var(--accent); border-radius: 50%;
  animation: spin 0.8s linear infinite; margin: 0 auto 14px;
}
@keyframes spin { to { transform: rotate(360deg); } }
.progress-text { font-size: 0.95rem; color: var(--muted); margin-top: 10px; }
.progress-sub { font-size: 0.8rem; color: #64748b; margin-top: 6px; }

.card {
  background: var(--card); border: 1px solid var(--border);
  border-radius: 12px; padding: 14px; margin-bottom: 12px;
}
.match-title { font-size: 1.05rem; font-weight: 600; margin-bottom: 2px; }
.match-meta { font-size: 0.78rem; color: var(--muted); margin-bottom: 10px; }
.lean-banner {
  display: inline-block; padding: 6px 14px; border-radius: 8px;
  font-weight: 700; font-size: 0.95rem; margin-bottom: 12px;
}
.lean-home { background: #14532d; color: #4ade80; border: 1px solid #22c55e; }
.lean-away { background: #7f1d1d; color: #fca5a5; border: 1px solid #ef4444; }
.lean-draw { background: #1e3a5f; color: #93c5fd; border: 1px solid #3b82f6; }
.section { margin-bottom: 9px; }
.section-label { font-size: 0.7rem; color: var(--muted); text-transform: uppercase; margin-bottom: 4px; }
.row { display: flex; flex-wrap: wrap; gap: 6px; }
.pill { background: #1f2937; padding: 4px 9px; border-radius: 7px; font-size: 0.88rem; }
.pill.strong { background: #14532d; color: #4ade80; font-weight: 700; border: 1px solid #22c55e; }
.pill.strong-red { background: #7f1d1d; color: #fca5a5; font-weight: 700; border: 1px solid #ef4444; }
input[type=text], textarea {
  width: 100%; padding: 11px; border-radius: 9px; border: 1px solid var(--border);
  background: #1f2937; color: white; font-size: 0.95rem; margin-bottom: 8px;
}
textarea { min-height: 110px; resize: vertical; }
button {
  width: 100%; padding: 12px; border: none; border-radius: 9px;
  background: #0ea5e9; color: white; font-size: 0.95rem; font-weight: 600; cursor: pointer;
}
.btn-secondary { background: #334155; }
.paste-stats {
  font-size: 0.85rem; color: var(--muted); margin: 8px 0 12px;
  padding: 8px 12px; background: #1e293b; border-radius: 8px;
}
.coming-soon { text-align: center; padding: 40px 20px; color: var(--muted); }
</style>
</head>
<body>
<div class="header">
  <h1>⚽ Sports Predict</h1>
  <div class="status" id="status">{{ status }}</div>
</div>

<div class="tabs">
  <button class="tab active" onclick="showTab('soccer')">Soccer</button>
  <button class="tab" onclick="showTab('basketball')">Basketball</button>
  <button class="tab" onclick="showTab('nfl')">NFL</button>
  <button class="tab" onclick="showTab('baseball')">Baseball</button>
  <button class="tab" onclick="showTab('tennis')">Tennis</button>
</div>

<div id="soccer" class="tab-content active">

  {% if not ready %}
  <div class="loading-box" id="loading-box">
    <div class="spinner"></div>
    <div style="font-weight:600">Model is training...</div>
    <div class="progress-text" id="progress-text">{{ progress }}</div>
    <div class="progress-sub">This page will update automatically when ready.<br>Please keep this tab open.</div>
  </div>
  {% endif %}

  <div class="card">
    <form method="POST">
      <input type="hidden" name="action" value="single">
      <input type="text" name="home" placeholder="Home team" value="{{ home or '' }}" {% if not ready %}disabled{% endif %}>
      <input type="text" name="away" placeholder="Away team" value="{{ away or '' }}" {% if not ready %}disabled{% endif %}>
      <button type="submit" {% if not ready %}disabled{% endif %}>Predict Match</button>
    </form>

    <form method="POST" style="margin-top:14px">
      <input type="hidden" name="action" value="paste">
      <textarea name="paste_list" placeholder="Paste any list of matches here..." {% if not ready %}disabled{% endif %}>{{ paste_text or '' }}</textarea>
      <button type="submit" class="btn-secondary" {% if not ready %}disabled{% endif %}>Predict Pasted List</button>
    </form>

    {% if paste_stats %}
    <div class="paste-stats">{{ paste_stats }}</div>
    {% endif %}

    {% if result %}
    <div style="margin-top:14px">
      <div class="match-title">{{ result.home }} vs {{ result.away }}</div>
      <div class="lean-banner {{ result.lean_class }}">LEAN {{ result.lean }} {{ result.lean_pct }}%</div>
      <div class="section">
        <div class="section-label">1X2</div>
        <div class="row">
          <span class="pill {% if result.strong_home %}strong{% endif %}">H {{ result.home_win }}%</span>
          <span class="pill">D {{ result.draw }}%</span>
          <span class="pill {% if result.strong_away %}strong-red{% endif %}">A {{ result.away_win }}%</span>
        </div>
      </div>
      <div class="section">
        <div class="section-label">Over / Under 2.5</div>
        <div class="row">
          <span class="pill {% if result.strong_over %}strong{% endif %}">Over {{ result.over25 }}%</span>
          <span class="pill {% if result.strong_under %}strong{% endif %}">Under {{ result.under25 }}%</span>
        </div>
      </div>
      <div class="section">
        <div class="section-label">Both Teams To Score</div>
        <div class="row">
          <span class="pill {% if result.strong_btts_yes %}strong{% endif %}">Yes {{ result.btts_yes }}%</span>
          <span class="pill {% if result.strong_btts_no %}strong{% endif %}">No {{ result.btts_no }}%</span>
        </div>
      </div>
    </div>
    {% endif %}

    {% if paste_results %}
    <div style="margin-top:10px">
      {% for p in paste_results %}
      <div class="card" style="background:#0f172a;margin-bottom:8px">
        <div class="match-title">{{ p.home }} vs {{ p.away }}</div>
        <div class="match-meta">{{ p.league }}</div>
        <div class="lean-banner {{ p.lean_class }}" style="font-size:0.85rem;padding:4px 10px">LEAN {{ p.lean }} {{ p.lean_pct }}%</div>
        <div class="row" style="margin-top:6px">
          <span class="pill">H {{ p.home_win }}%</span>
          <span class="pill">D {{ p.draw }}%</span>
          <span class="pill">A {{ p.away_win }}%</span>
          <span class="pill">U {{ p.under25 }}%</span>
        </div>
      </div>
      {% endfor %}
    </div>
    {% endif %}

    {% if error %}<div style="color:#f87171;margin-top:10px">{{ error }}</div>{% endif %}
  </div>

  {% if today_preds %}
    {% for p in today_preds %}
    <div class="card">
      <div class="match-title">{{ p.home }} vs {{ p.away }}</div>
      <div class="match-meta">{{ p.match_date }} • {{ p.league }} • {{ p.time }}</div>
      <div class="lean-banner {{ p.lean_class }}">LEAN {{ p.lean }} {{ p.lean_pct }}%</div>
      <div class="section">
        <div class="section-label">1X2</div>
        <div class="row">
          <span class="pill {% if p.strong_home %}strong{% endif %}">H {{ p.home_win }}%</span>
          <span class="pill">D {{ p.draw }}%</span>
          <span class="pill {% if p.strong_away %}strong-red{% endif %}">A {{ p.away_win }}%</span>
        </div>
      </div>
    </div>
    {% endfor %}
  {% endif %}
</div>

<div id="basketball" class="tab-content"><div class="coming-soon"><h2>🏀 Basketball</h2><p>Coming soon</p></div></div>
<div id="nfl" class="tab-content"><div class="coming-soon"><h2>🏈 NFL</h2><p>Coming soon</p></div></div>
<div id="baseball" class="tab-content"><div class="coming-soon"><h2>⚾ Baseball</h2><p>Coming soon</p></div></div>
<div id="tennis" class="tab-content"><div class="coming-soon"><h2>🎾 Tennis</h2><p>Coming soon</p></div></div>

<script>
function showTab(id) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  event.target.classList.add('active');
}

{% if not ready %}
let checkCount = 0;
function checkStatus() {
  checkCount++;
  fetch('/status')
    .then(r => r.json())
    .then(data => {
      document.getElementById('status').textContent = data.message;
      const prog = document.getElementById('progress-text');
      if (prog) prog.textContent = data.progress || data.message;
      if (data.ready) {
        location.reload();
      } else {
        setTimeout(checkStatus, 2500);
      }
    })
    .catch(() => setTimeout(checkStatus, 4000));
}
setTimeout(checkStatus, 2000);
{% endif %}
</script>
</body>
</html>
"""

@app.route("/status")
def status():
    return jsonify({
        "ready": MODEL["ready"],
        "message": MODEL["message"],
        "progress": MODEL.get("progress", "")
    })

@app.route("/", methods=["GET", "POST"])
def index():
    result = error = None
    home = away = paste_text = paste_stats = ""
    paste_results = []

    if request.method == "POST" and MODEL["ready"]:
        action = request.form.get("action", "single")
        if action == "single":
            home = request.form.get("home", "").strip()
            away = request.form.get("away", "").strip()
            pred = predict(home, away)
            if pred:
                result = pred
            else:
                error = f"Could not match teams: {home} / {away}"
        elif action == "paste":
            paste_text = request.form.get("paste_list", "")
            extracted = extract_matches_from_text(paste_text)
            success = 0
            skipped = 0
            for item in extracted:
                p = predict(item["home"], item["away"])
                if p:
                    p["league"] = item.get("league", "")
                    paste_results.append(p)
                    success += 1
                else:
                    skipped += 1
            paste_stats = f"Found {len(extracted)} possible matches → {success} predicted, {skipped} skipped (unknown teams)"
            if not paste_results and extracted:
                error = "None of the extracted teams were found in the model"
            elif not extracted:
                error = "Could not extract any matches from the pasted text"

    return render_template_string(HTML,
        status=MODEL["message"], progress=MODEL.get("progress", ""), ready=MODEL["ready"],
        result=result, error=error, home=home, away=away,
        paste_text=paste_text, paste_results=paste_results, paste_stats=paste_stats,
        today_preds=MODEL["today_preds"])

if __name__ == "__main__":
    print("Open → http://localhost:8801")
    app.run(host="0.0.0.0", port=8801, debug=False)
