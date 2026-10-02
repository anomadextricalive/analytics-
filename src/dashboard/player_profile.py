"""Player profile for the Player Explorer deep dive: photo, bio facts and ESPN Cricinfo career lines.

Everything comes from the bundled SQLite DB except the headshot, which is a public ESPN CDN image looked up by
ESPN Cricinfo id (checked once per player, cached; an initials avatar is used when there is no photo).
"""
import html
from datetime import date, datetime
from urllib.request import Request, urlopen

import streamlit as st

HEADSHOT = "https://a.espncdn.com/i/headshots/cricket/players/full/{eid}.png"
PROFILE = "https://www.cricinfo.com/ci/content/player/{eid}.html"
_BAT = {"RHB": "Right-hand bat", "LHB": "Left-hand bat", "Right-hand bat": "Right-hand bat", "Left-hand bat": "Left-hand bat"}


@st.cache_data(show_spinner=False, ttl=7 * 24 * 3600)
def headshot_url(espn_id: str):
    """The ESPN headshot URL if the image exists, else None."""
    url = HEADSHOT.format(eid=espn_id)
    try:
        with urlopen(Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"}), timeout=4) as r:
            return url if r.status == 200 and "image" in r.headers.get("Content-Type", "") else None
    except Exception:
        return None


def _initials(name: str) -> str:
    parts = [w for w in str(name).replace(".", " ").split() if w]
    return (parts[0][0] + (parts[-1][0] if len(parts) > 1 else "")).upper() if parts else "?"


def _dob(dob):
    if not dob or str(dob) in ("None", "NaT", "nan"):
        return None
    try:
        d = datetime.strptime(str(dob)[:10], "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    age = date.today().year - d.year - ((date.today().month, date.today().day) < (d.month, d.day))
    return f"{d.day} {d.strftime('%b %Y')} (age {age})"


def _num(v, nd=0):
    try:
        f = float(v)
        return "-" if f != f else (f"{f:,.{nd}f}" if nd else f"{int(round(f)):,}")
    except (TypeError, ValueError):
        return "-"


def load_extras(query_fn, pid: int) -> dict:
    """ESPN id, our own coverage of the player, and the ESPN career lines. Each piece degrades to empty on failure."""
    out = {"espn_id": None, "first": None, "last": None, "matches": 0, "teams": [], "comps": [], "career": {}}
    m = query_fn("SELECT espn_id FROM player_espn_map WHERE player_id = :pid AND status = 'matched'", pid=pid)
    if not m.empty and m.iloc[0]["espn_id"] is not None:
        out["espn_id"] = str(m.iloc[0]["espn_id"]).split(".")[0]
    cov = query_fn("""
        SELECT MIN(d) AS first, MAX(d) AS last, COUNT(DISTINCT mid) AS matches FROM (
            SELECT m.match_date AS d, m.id AS mid FROM player_innings pi JOIN matches m ON m.id = pi.match_id WHERE pi.batter_id = :pid
            UNION ALL
            SELECT m.match_date, m.id FROM player_bowling_innings pb JOIN matches m ON m.id = pb.match_id WHERE pb.bowler_id = :pid)
    """, pid=pid)
    if not cov.empty and cov.iloc[0]["matches"]:
        out.update(first=str(cov.iloc[0]["first"])[:10], last=str(cov.iloc[0]["last"])[:10], matches=int(cov.iloc[0]["matches"]))
    teams = query_fn("""
        SELECT t.name AS name, COUNT(*) AS n FROM (
            SELECT i.batting_team_id AS tid FROM player_innings pi JOIN innings i ON i.id = pi.innings_id WHERE pi.batter_id = :pid
            UNION ALL
            SELECT i.bowling_team_id FROM player_bowling_innings pb JOIN innings i ON i.id = pb.innings_id WHERE pb.bowler_id = :pid) x
        JOIN teams t ON t.id = x.tid GROUP BY t.name ORDER BY n DESC LIMIT 6
    """, pid=pid)
    out["teams"] = teams["name"].tolist() if not teams.empty else []
    comps = query_fn("""
        SELECT COALESCE(tr.display_name, m.tournament) AS name, COUNT(DISTINCT m.id) AS n FROM (
            SELECT match_id FROM player_innings WHERE batter_id = :pid
            UNION SELECT match_id FROM player_bowling_innings WHERE bowler_id = :pid) x
        JOIN matches m ON m.id = x.match_id LEFT JOIN tournaments tr ON tr.code = m.tournament
        GROUP BY name ORDER BY n DESC LIMIT 6
    """, pid=pid)
    out["comps"] = comps["name"].tolist() if not comps.empty else []
    if out["espn_id"]:
        c = query_fn("""
            SELECT fmt, stat_type, span, mat, inns, runs, bf, hs, ave, sr, hundreds, fifties, fours, sixes,
                   wkts, econ, bbi, balls FROM espn_career WHERE espn_id = :eid AND status = 'ok'
        """, eid=out["espn_id"])
        for r in c.to_dict("records"):
            out["career"].setdefault(r["fmt"], {})["bat" if r["stat_type"] == "batting" else "bowl"] = r
    return out


def hero_html(p: dict, display_name: str, espn_id, photo) -> str:
    """Name, photo and chips. Photo falls back to an initials avatar."""
    esc = lambda s: html.escape(str(s))
    chips = []
    if p.get("country"): chips.append(f'<span class="pe-chip">🏏 {esc(p["country"])}</span>')
    if p.get("player_role"): chips.append(f'<span class="pe-chip pe-chip-role"><b>{esc(p["player_role"])}</b></span>')
    if p.get("batting_style"): chips.append(f'<span class="pe-chip">{esc(_BAT.get(p["batting_style"], p["batting_style"]))}</span>')
    if p.get("bowling_style"): chips.append(f'<span class="pe-chip">{esc(p["bowling_style"])}</span>')
    d = _dob(p.get("date_of_birth"))
    if d: chips.append(f'<span class="pe-chip">b. {esc(d)}</span>')
    key = p.get("name", "")
    sub = f'<div class="pe-hero-sub">{esc(key)}</div>' if key and key != display_name else ""
    avatar = (f'<img class="pe-avatar" src="{esc(photo)}" alt="{esc(display_name)}">' if photo
              else f'<div class="pe-avatar pe-avatar-fallback">{esc(_initials(display_name))}</div>')
    link = (f'<a class="pe-link" href="{PROFILE.format(eid=esc(espn_id))}" target="_blank" rel="noopener">ESPN Cricinfo profile ↗</a>'
            if espn_id else "")
    return (f'<div class="pe-hero pe-hero-flex">{avatar}<div class="pe-hero-body"><div class="pe-hero-name">{esc(display_name)}</div>'
            f'{sub}<div class="pe-chips">{"".join(chips)}</div>{link}</div></div>')


def _facts(p: dict, ex: dict) -> str:
    rows = [("Full name", p.get("full_name")), ("Born", _dob(p.get("date_of_birth"))), ("Country", p.get("country")),
            ("Role", p.get("player_role")), ("Batting", _BAT.get(p.get("batting_style"), p.get("batting_style"))),
            ("Bowling", p.get("bowling_style")),
            ("First match here", ex["first"]), ("Latest match here", ex["last"]),
            ("Teams", " · ".join(ex["teams"]) or None), ("Competitions", " · ".join(ex["comps"]) or None),
            ("ESPN Cricinfo id", ex["espn_id"])]
    body = "".join(f'<tr><td class="pp-k">{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>'
                   for k, v in rows if v not in (None, "", "None", "nan"))
    return f'<table class="pp-facts">{body}</table>'


def render_profile(query_fn, p: dict, display_name: str):
    """Draw the hero card (photo, chips, ESPN link), then the bio table and ESPN career block."""
    ex = load_extras(query_fn, int(p["id"]))
    st.markdown(hero_html(p, display_name, ex["espn_id"], headshot_url(ex["espn_id"]) if ex["espn_id"] else None),
                unsafe_allow_html=True)
    left, right = st.columns([5, 7])
    with left:
        st.markdown('<div class="nb-label">Bio</div>', unsafe_allow_html=True)
        st.markdown(_facts(p, ex), unsafe_allow_html=True)
    with right:
        st.markdown('<div class="nb-label">Career · ESPN Cricinfo (every T20 played)</div>', unsafe_allow_html=True)
        if not ex["career"]:
            st.caption("No ESPN Cricinfo career line is linked to this player yet.")
            return
        fmts = [f for f in ("t20", "t20i") if f in ex["career"]]
        for tab, f in zip(st.tabs([{"t20": "All T20", "t20i": "T20I"}[f] for f in fmts]), fmts):
            with tab:
                bat, bowl = ex["career"][f].get("bat"), ex["career"][f].get("bowl")
                if bat and bat.get("inns"):
                    c = st.columns(6)
                    for col, (lab, v) in zip(c, [("Matches", _num(bat["mat"])), ("Runs", _num(bat["runs"])), ("Average", _num(bat["ave"], 2)),
                                                 ("Strike rate", _num(bat["sr"], 1)), ("High score", bat["hs"] or "-"),
                                                 ("100s / 50s", f'{_num(bat["hundreds"])} / {_num(bat["fifties"])}')]):
                        col.metric(lab, v)
                if bowl and bowl.get("wkts"):
                    c = st.columns(6)
                    for col, (lab, v) in zip(c, [("Bowled in", _num(bowl["inns"])), ("Wickets", _num(bowl["wkts"])), ("Economy", _num(bowl["econ"], 2)),
                                                 ("Average", _num(bowl["ave"], 2)), ("Best", bowl["bbi"] or "-"),
                                                 ("Strike rate", _num(bowl["sr"], 1))]):
                        col.metric(lab, v)
                span = (bat or bowl or {}).get("span")
                if f == "t20" and (bat or bowl):
                    mat = (bat or bowl).get("mat")
                    st.caption(f"Span {span or '-'} · ESPN Cricinfo counts {_num(mat)} T20 matches; this database has {ex['matches']:,} match(es) with this player batting or bowling.")
                elif span:
                    st.caption(f"Span {span}")
