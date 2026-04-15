"""
Cricket Insight Engine — Text-to-SQL + Empirical Fact Synthesis
---------------------------------------------------------------
Pipeline:
  1. Receive a natural-language question
  2. Ask the LLM to generate 3–5 targeted SQL queries that surface records,
     extremes, and anomalies (not pre-aggregated summaries)
  3. Execute those queries safely against the local SQLite database
  4. Feed raw numeric results back to the LLM and ask it to identify
     the single most surprising empirical fact

The key difference from retrieval-based approaches: every answer is grounded
in actual row-level data pulled at query time — specific match dates, exact
counts, named players, exact figures.
"""

import hashlib
import json
import os
import re
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import pandas as pd
import requests

# ─────────────────────────────────────────────────────────────────────────────
# IN-PROCESS RESULT CACHE
# Cache key: SHA-256(question + db_mtime).  TTL: 1 hour.
# The DB is static between pipeline runs, so repeated questions are free.
# ─────────────────────────────────────────────────────────────────────────────
_cache: dict[str, tuple[float, dict]] = {}   # key → (expires_at, result)
_CACHE_TTL = 3600  # seconds


def _cache_key(question: str, db_path) -> str:
    mtime = str(os.path.getmtime(str(db_path))) if os.path.exists(str(db_path)) else "0"
    raw = f"{question.strip().lower()}|{mtime}"
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> Optional[dict]:
    entry = _cache.get(key)
    if entry and time.time() < entry[0]:
        return entry[1]
    return None


def _cache_set(key: str, value: dict) -> None:
    _cache[key] = (time.time() + _CACHE_TTL, value)

# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA CONTEXT  (compact but complete — all 26 tables)
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA_CONTEXT = """
SQLite T20 cricket database. 5,811 matches · 1.3 M deliveries · 2004–2026.
phase encoding: 0=powerplay (overs 0–5)  1=middle (6–14)  2=death (15–19)
over_number is 0-indexed (0 = first over, 19 = last over).
tournaments: 't20i_male', 'T20 World Cup', 'IPL', 'PSL', 'BBL', 'CPL',
             'T20 Blast', 'SA20', 'LPL', 'ILT20', 'The Hundred'
venue.country exact values (use these verbatim in WHERE clauses):
  'India', 'Australia', 'UK', 'South Africa', 'New Zealand', 'UAE',
  'Bangladesh', 'Pakistan', 'Sri Lanka', 'Ireland', 'Scotland', 'USA'
  IMPORTANT: England/Wales/Britain are stored as 'UK' — never use 'England' or 'United Kingdom'

TABLE COLUMNS (→ denotes FK target):
players(id,
        cricsheet_key,   -- ← USE THIS as the player name (e.g. 'V Kohli', 'RG Sharma')
        full_name,       -- always NULL, do not use
        cricsheet_uuid, country, batting_style, bowling_style,
        t20_bat_innings, t20_bat_runs, t20_bat_balls, t20_bowl_wickets,
        t20_field_catches)

venues(id, name, city, country, boundary_straight_m, boundary_square_m,
       avg_first_inn_score, avg_second_inn_score, pace_index, boundary_ease,
       venue_bat_factor, venue_bowl_factor)

matches(id, cricsheet_id, match_date, tournament, season,
        venue_id→venues, team1_id→teams, team2_id→teams,
        toss_winner_id→teams, toss_decision,
        winner_id→teams, win_by_runs, win_by_wickets, chasing_team_won,
        no_result)

teams(id, name)

innings(id, match_id→matches, innings_number, batting_team_id→teams,
        bowling_team_id→teams, target, total_runs, total_wickets,
        total_balls, extras)

deliveries(id, innings_id→innings, over_number, ball_in_over,
           ball_number, phase,
           batter_id→players, non_striker_id→players, bowler_id→players,
           bat_runs, extras, total_runs,
           is_boundary_4, is_boundary_6, is_dot,
           wide, no_ball, bye, leg_bye,
           is_wicket, wicket_kind, player_out_id→players,
           req_rate_at_ball, crr_at_ball)

player_innings(id, innings_id, match_id, batter_id→players, team_id,
               batting_position, runs, balls_faced, fours, sixes, not_out,
               dismissal_kind,
               pp_runs, pp_balls, mid_runs, mid_balls, death_runs, death_balls,
               is_chase, chase_won, required_rr_start)

player_bowling_innings(id, innings_id, match_id, bowler_id→players, team_id,
                       overs_bowled, balls_bowled, runs_conceded, wickets,
                       maidens, dot_balls, wides, no_balls, boundaries_conceded,
                       pp_balls, pp_runs, mid_balls, mid_runs,
                       death_balls, death_runs)

player_career_bat(player_id PK, tournament PK,
                  innings, not_outs, runs, balls, hs, thirties, fifties,
                  hundreds, ducks, fours, sixes, median_score,
                  times_opened, top_scored, average, strike_rate,
                  pp_sr, mid_sr, death_sr, adj_average, adj_strike_rate)

player_career_bowl(player_id PK, tournament PK,
                   innings, balls, runs, wickets, dot_balls, economy,
                   average, strike_rate, dot_pct,
                   pp_economy, mid_economy, death_economy, adj_economy)

player_position_bat(player_id PK, position PK,
                    innings, not_outs, runs, balls, average, strike_rate,
                    pp_sr, mid_sr, death_sr)

player_chase_bat(player_id PK, innings_type PK,   -- 'chase' or 'first'
                 innings, not_outs, runs, balls, average, strike_rate,
                 high_pressure_innings, high_pressure_sr, finishes)

player_venue_bat(player_id PK, venue_id PK,
                 innings, not_outs, runs, balls, average, strike_rate)

player_venue_bowl(player_id PK, venue_id PK,
                  innings, balls, runs, wickets, economy, average)

player_perf_by_opponent(player_id PK, opponent_id PK,
                        bat_innings, bat_not_outs, bat_runs, bat_balls,
                        bat_hs, bat_fifties, bat_hundreds, bat_ducks,
                        bat_average, bat_sr,
                        bowl_innings, bowl_balls, bowl_runs, bowl_wickets,
                        bowl_economy, bowl_average, bowl_sr)

player_perf_by_season(player_id PK, season PK, tournament PK,
                      bat_innings, bat_not_outs, bat_runs, bat_balls,
                      bat_hs, bat_fifties, bat_hundreds, bat_ducks,
                      bat_average, bat_sr,
                      bowl_innings, bowl_balls, bowl_runs, bowl_wickets,
                      bowl_economy, bowl_average)

player_perf_by_team(player_id PK, team_id PK,
                    bat_innings, bat_runs, bat_average, bat_sr,
                    bowl_innings, bowl_wickets, bowl_economy)

player_perf_by_result(player_id PK, result PK,   -- 'won', 'lost', 'no_result'
                      bat_innings, bat_not_outs, bat_runs, bat_balls,
                      bat_hs, bat_fifties, bat_ducks,
                      bat_average, bat_sr,
                      bowl_innings, bowl_balls, bowl_runs, bowl_wickets,
                      bowl_economy)

player_dismissal_analysis(player_id PK, dismissal_kind PK, count, pct)

player_bowling_dismissal_analysis(player_id PK, dismissal_kind PK, count, pct)

player_fielding_stats(player_id PK, tournament PK,
                      catches, run_outs_direct, run_outs_assisted, stumpings,
                      most_catches_inn)

player_milestones(id, player_id→players, match_id→matches, innings_id,
                  milestone_type,   -- 'fifty','hundred','thirty','duck',
                                    -- 'four_wicket_haul','five_wicket_haul',
                                    -- 'player_of_match'
                  value, opposition_id→teams, venue_id→venues,
                  match_date, tournament)

player_of_match_awards(id, player_id→players, match_id→matches,
                       match_date, tournament,
                       opposition_id→teams, venue_id→venues)

venue_difficulty(venue_id PK, total_matches,
                 avg_first_inn_runs, avg_second_inn_runs, avg_total_runs,
                 avg_wickets_per_inn, bat_factor, bowl_factor,
                 pace_index, spin_index, boundary_rate,
                 bat_factor_lo, bat_factor_hi)

player_ratings(player_id PK, tournament PK,
               bat_rating, bowl_rating, overall_rating,
               opener_score, finisher_score, anchor_score,
               pp_bat_score, death_bat_score, chase_score,
               pp_bowl_score, mid_bowl_score, death_bowl_score,
               adj_bat_rating, adj_bowl_rating, updated_at)

player_similarity(player_id_a PK, player_id_b PK, tournament PK, similarity)

player_form(player_id PK,
            avg_5, avg_10, avg_20, sr_5, sr_10, sr_20,
            career_avg, career_sr, cv,
            breakout_flag, breakout_delta, innings_total, updated_at)

partnerships(id, innings_id, match_id,
             wicket_number, batter1_id→players, batter2_id→players,
             runs, balls, batter1_runs, batter2_runs)
"""

# ─────────────────────────────────────────────────────────────────────────────
# PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

_SQL_GEN_SYSTEM = """You are a cricket analytics SQL expert with access to the SQLite schema below.

Given a natural-language question, generate 3–5 SQLite SELECT queries that surface
EMPIRICAL, SURPRISING facts — not just career averages. Think: records, extremes,
paradoxes, cross-dimensional anomalies, era-based patterns.

Good query targets:
  - All-time or era records with exact match context (date, tournament, opponent)
  - Largest gaps between a player's career norm and performance in a specific
    context (venue / opponent / phase / result)
  - Counts that reveal hidden patterns (dot balls per match, sixes in a season,
    most wickets in powerplay across all tournaments)
  - Paradoxes: great averages paired with terrible strike rate, or vice versa
  - Venue facts: toss advantage, chasing win %, average 1st-innings score vs
    global average, highest/lowest scoring grounds

Rules:
  - ONLY SELECT statements
  - Always JOIN to get human-readable names: players.cricsheet_key (NOT full_name —
    it is always NULL), venues.name, teams.name, matches.match_date, matches.tournament
  - LIMIT each query to 10 rows
  - Use ROUND(x, 2) on decimals
  - Filter out tiny sample sizes (e.g. WHERE innings >= 5 or balls >= 60)
  - Do NOT use CTEs with WITH … AS if avoidable — use subqueries instead

Return ONLY valid JSON in this exact format (no markdown, no extra text):
{
  "entity_type": "player|venue|tournament|era|general",
  "entity_name": "resolved name of the subject",
  "what_to_find": "one sentence — what kind of anomaly or record to surface",
  "queries": [
    {"label": "short description", "sql": "SELECT ..."},
    ...
  ]
}"""

_SYNTHESIS_SYSTEM = """You are a cricket analytics narrator. You receive raw query results from a
real T20 cricket database (5,811 matches, 1.3 M deliveries, 2004–2026).

Task: Identify the single most surprising, counter-intuitive, or record-breaking
empirical finding in the data and explain it in 2–4 punchy sentences.

Constraints:
  - Every claim must use EXACT numbers from the results (no vague language)
  - Include specific context: tournament names, seasons, dates, sample sizes
  - Write as a cricket analyst making a point that stops a conversation
  - Do not preface with "Based on the data…" or "According to the results…"
  - If multiple striking facts exist, pick the most extreme one
  - If results are sparse or empty, say so plainly and suggest a rephrased question"""


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

_BLOCKED_KEYWORDS = {
    "DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER",
    "ATTACH", "DETACH", "PRAGMA", "VACUUM", "REINDEX",
}


def _is_safe_sql(sql: str) -> bool:
    """Accept only SELECT statements with no destructive keywords."""
    clean = sql.strip().upper()
    if not clean.startswith("SELECT"):
        return False
    for kw in _BLOCKED_KEYWORDS:
        # Word-boundary check so we don't block column names like "updated_at"
        if re.search(rf"\b{kw}\b", clean):
            return False
    return True


def _extract_json(text: str) -> Optional[dict]:
    """Parse JSON from LLM output that may be wrapped in markdown fences."""
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{[\s\S]+\}", text)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass
    return None


def _call_gemini(
    api_key: str,
    model: str,
    system_prompt: str,
    user_message: str,
    temperature: float = 0.1,
) -> tuple[str, str]:
    """Single Gemini API call. Returns (text, status)."""
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/"
        f"models/{model}:generateContent?key={api_key}"
    )
    payload = {
        "system_instruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": user_message}]}],
        "generationConfig": {"temperature": temperature},
    }
    try:
        res = requests.post(url, json=payload, timeout=60)
        if res.status_code >= 300:
            return "", f"http_{res.status_code}"
        data = res.json()
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                t = part.get("text", "")
                if t.strip():
                    return t.strip(), "ok"
        return "", "empty_response"
    except requests.Timeout:
        return "", "timeout"
    except Exception as exc:
        return "", f"exception: {exc}"


# ─────────────────────────────────────────────────────────────────────────────
# CORE FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def resolve_entities(question: str, db_path) -> str:
    """
    Fuzzy-match any venue / player / team names mentioned in the question
    against the actual DB. Returns a compact context block injected into
    the SQL-generation prompt so the LLM uses exact names and knows data
    volume before writing queries.

    Data tiers:
      RICH   >= 20 matches   — full analysis possible
      SPARSE  5–19 matches   — limited patterns, warn user
      THIN    1–4  matches   — almost no data, say so explicitly
    """
    q_lower = question.lower()
    lines = []

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        # ── Venues ──────────────────────────────────────────────────────────
        venues = conn.execute(
            "SELECT v.id, v.name, v.city, v.country, COUNT(m.id) as matches "
            "FROM venues v LEFT JOIN matches m ON m.venue_id = v.id "
            "GROUP BY v.id"
        ).fetchall()

        for vid, vname, vcity, vcountry, vmatch in venues:
            tokens = [t for t in (vname or "").lower().split()
                      if len(t) >= 3 and t not in {"the", "and", "of", "at"}]
            city_tok = (vcity or "").lower()
            if any(t in q_lower for t in tokens) or (city_tok and city_tok in q_lower):
                tier = "RICH" if vmatch >= 20 else "SPARSE" if vmatch >= 5 else "THIN"
                lines.append(
                    f"VENUE MATCH  exact_name='{vname}'  city={vcity}  "
                    f"country={vcountry}  matches={vmatch}  data_tier={tier}"
                )
                if tier == "THIN":
                    lines.append(
                        f"  ⚠ Only {vmatch} match(es) at '{vname}' — "
                        "do not attempt phase/trend analysis; state data is insufficient."
                    )
                elif tier == "SPARSE":
                    lines.append(
                        f"  ⚠ Only {vmatch} matches at '{vname}' — "
                        "limit to simple aggregates; note small sample in synthesis."
                    )

        # ── Players ─────────────────────────────────────────────────────────
        players = conn.execute(
            "SELECT id, cricsheet_key, country FROM players"
        ).fetchall()

        for pid, pkey, pcountry in players:
            if not pkey:
                continue
            key_lower = pkey.lower()
            # Match if any surname token (≥4 chars) appears in question
            parts = [p for p in key_lower.split() if len(p) >= 4]
            if any(p in q_lower for p in parts) or key_lower in q_lower:
                # Get innings count as proxy for data richness
                bat_inn = conn.execute(
                    "SELECT innings FROM player_career_bat WHERE player_id=? AND tournament='ALL'",
                    (pid,)
                ).fetchone()
                innings = bat_inn[0] if bat_inn else 0
                tier = "RICH" if innings >= 20 else "SPARSE" if innings >= 5 else "THIN"
                lines.append(
                    f"PLAYER MATCH  exact_key='{pkey}'  player_id={pid}  "
                    f"country={pcountry}  bat_innings={innings}  data_tier={tier}"
                )

        # ── Teams ────────────────────────────────────────────────────────────
        teams = conn.execute("SELECT id, name FROM teams").fetchall()
        for tid, tname in teams:
            if tname and tname.lower() in q_lower:
                lines.append(f"TEAM MATCH  exact_name='{tname}'  team_id={tid}")

    finally:
        conn.close()

    if not lines:
        return ""
    return (
        "\n\nENTITY RESOLUTION (use these exact names/ids in your SQL — do not guess):\n"
        + "\n".join(lines)
    )


def _run_one_query(args: tuple) -> dict:
    """Execute a single query. Called from a thread pool."""
    db_path, label, sql = args
    if not _is_safe_sql(sql):
        return {"label": label, "sql": sql, "data": [], "error": "Blocked: only SELECT is allowed"}
    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        df = pd.read_sql_query(sql, conn)
        conn.close()
        return {"label": label, "sql": sql, "data": df.head(10).to_dict(orient="records"), "error": None}
    except Exception as exc:
        return {"label": label, "sql": sql, "data": [], "error": str(exc)}


def execute_queries(db_path, queries: list[dict]) -> list[dict]:
    """
    Run a list of {"label": str, "sql": str} dicts against the SQLite DB
    in parallel using a thread pool. Each query gets its own connection
    (required for thread safety with SQLite).
    Returns list of {"label", "sql", "data": list[dict], "error": str|None}
    in the same order as the input.
    """
    args = [(db_path, q.get("label", "query"), q.get("sql", "").strip()) for q in queries]
    with ThreadPoolExecutor(max_workers=min(len(args), 5)) as pool:
        results = list(pool.map(_run_one_query, args))
    return results


def generate_insight(
    question: str,
    db_path,
    api_key: str,
    model: str = "gemini-2.0-flash",
) -> dict:
    """
    Full pipeline: question → SQL generation → execution → empirical insight.

    Returns dict with keys:
      insight       str | None    — the synthesized finding
      entity        str           — resolved entity name (player / venue / etc.)
      what_to_find  str           — what kind of anomaly was targeted
      queries_run   list[dict]    — the generated queries (label + sql)
      raw_results   list[dict]    — executed results (label + sql + data + error)
      cached        bool          — True if result came from cache
      error         str | None    — top-level error if pipeline failed early
    """
    # ── Cache check ──────────────────────────────────────────────────────────
    ck = _cache_key(question, db_path)
    cached = _cache_get(ck)
    if cached:
        return {**cached, "cached": True}

    # ── Step 1: Resolve entities against real DB ────────────────────────────
    entity_ctx = resolve_entities(question, db_path)

    # ── Step 2: Generate SQL queries ────────────────────────────────────────
    sql_prompt = f"Question: {question}\n\nSchema:\n{SCHEMA_CONTEXT}{entity_ctx}"
    raw_plan, status = _call_gemini(api_key, model, _SQL_GEN_SYSTEM, sql_prompt, temperature=0.1)

    if status != "ok":
        return {
            "insight": None, "entity": "", "what_to_find": "",
            "queries_run": [], "raw_results": [],
            "error": f"SQL generation failed ({status}). Check your API key or try again.",
        }

    plan = _extract_json(raw_plan)
    if not plan or not plan.get("queries"):
        return {
            "insight": None, "entity": "", "what_to_find": "",
            "queries_run": [], "raw_results": [],
            "error": "LLM returned an invalid query plan. Try rephrasing the question.",
        }

    queries = plan["queries"][:5]  # cap at 5

    # ── Step 3: Execute queries ──────────────────────────────────────────────
    raw_results = execute_queries(db_path, queries)

    # ── Step 4: Build synthesis context ─────────────────────────────────────
    result_blocks = []
    for r in raw_results:
        block = f"### {r['label']}\n"
        if r.get("error"):
            block += f"Error: {r['error']}\n"
        elif r["data"]:
            df = pd.DataFrame(r["data"])
            block += df.to_string(index=False, max_rows=10) + "\n"
        else:
            block += "(no rows returned)\n"
        result_blocks.append(block)

    synth_prompt = (
        f"Question asked: {question}\n\n"
        f"Data results:\n" + "\n".join(result_blocks)
    )

    insight, status2 = _call_gemini(
        api_key, model, _SYNTHESIS_SYSTEM, synth_prompt, temperature=0.35
    )

    if status2 != "ok":
        insight = (
            "Could not synthesize narrative — raw results are available below. "
            f"(status: {status2})"
        )

    result = {
        "insight": insight,
        "entity": plan.get("entity_name", ""),
        "what_to_find": plan.get("what_to_find", ""),
        "queries_run": queries,
        "raw_results": raw_results,
        "cached": False,
        "error": None,
    }
    _cache_set(ck, result)
    return result
