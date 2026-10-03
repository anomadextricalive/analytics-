"""Venue Predictor tab: pick a player and a stadium, see the likely score and why.

Backend and theory: src/analytics/venue_model.py, docs/VENUE_PREDICTOR_THEORY.md. The page states plainly how small a ground's effect is
next to the spread of a single innings."""
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import venue_model as V

_TRAIT_LABEL = {"elevation_m": "Altitude", "boundary_mean_m": "Boundary size", "boundary_asym_m": "Boundary asymmetry", "abs_lat": "Latitude",
                "capacity": "Capacity", "floodlights": "Floodlights", "pt_drop_in": "Drop-in pitch", "pt_synthetic": "Synthetic pitch"}


@st.cache_resource(show_spinner="Loading venue model…")
def _art(_sql, version: str):
    return V.load_artifacts(_sql)


@st.cache_data(show_spinner=False, ttl=3600)
def _players(_sql, version: str) -> pd.DataFrame:
    return _sql("""SELECT p.id, COALESCE(p.full_name, p.cricsheet_key) AS name, p.cricsheet_key AS key, p.country, COALESCE(b.n, 0) AS bat_n, COALESCE(w.n, 0) AS bowl_n
                   FROM players p LEFT JOIN (SELECT batter_id, COUNT(*) n FROM player_innings GROUP BY 1) b ON b.batter_id = p.id
                   LEFT JOIN (SELECT bowler_id, COUNT(*) n FROM player_bowling_innings GROUP BY 1) w ON w.bowler_id = p.id
                   WHERE COALESCE(b.n, 0) >= 15 OR COALESCE(w.n, 0) >= 15 ORDER BY name""")


@st.cache_data(show_spinner=False, ttl=3600)
def _tour_names(_sql) -> dict:
    t = _sql("SELECT code, display_name FROM tournaments")
    return {} if t.empty else {r.code: (r.display_name or r.code) for r in t.itertuples()}


def _venue_label(r) -> str:
    place = ", ".join(x for x in (r.get("city"), r.get("country")) if isinstance(x, str) and x)
    return f"{r['name']}" + (f" · {place}" if place and place not in r["name"] else "") + f"  ({int(r['n_innings'])} innings)"


def _fingerprint_chart(prof: pd.DataFrame, vid: int, plot_defaults):
    ref = prof[prof.n_innings >= 15]; row = prof[prof.venue_id == vid].iloc[0]; rows = []
    for col, lab in V.FINGERPRINT:
        if pd.isna(row.get(col)): continue
        sd = ref[col].std(); z = (row[col] - ref[col].mean()) / sd if sd else 0.0; rows.append((lab, float(np.clip(z, -3, 3)), row[col]))
    if not rows: return None
    fig = go.Figure(go.Bar(x=[r[1] for r in rows], y=[r[0] for r in rows], orientation="h", marker_color=["#FF4D8D" if r[1] < 0 else "#5EEAD4" for r in rows],
                           text=[f"{r[2]:+.1f}" if V.FINGERPRINT[i][0].startswith(("final", "eff")) else f"{r[2]:.0f}" for i, r in enumerate(rows)], textposition="outside",
                           hovertemplate="%{y}: %{x:+.1f} sd vs typical ground<extra></extra>"))
    fig.update_layout(height=290, margin=dict(l=0, r=30, t=10, b=10), xaxis=dict(range=[-3.4, 3.4], title="standard deviations from the typical ground", zeroline=True),
                      yaxis=dict(autorange="reversed"), showlegend=False)
    return plot_defaults(fig) if plot_defaults else fig


def render_venue_predictor(sql, plot_defaults=None, player_id=None, compact=False, key="vp"):
    art = _art(sql, "v1")
    if art is None:
        st.warning("The venue model has not been built yet. Run `python scripts/build_venue_model.py --build`."); return
    prof, meta, mdl = art; players = _players(sql, "v1")
    if players.empty: st.info("No players available."); return
    tour_names = _tour_names(sql)

    # ── pickers ──
    default_pid = player_id or st.session_state.get("vp_player_id")
    ids = players["id"].astype(int).tolist(); idx = ids.index(int(default_pid)) if default_pid is not None and int(default_pid) in ids else 0
    if compact:
        row = players[players.id == ids[idx]].iloc[0]; pid = int(row["id"]); st.caption(f"Player: **{row['name']}**")
    else:
        c1, c2 = st.columns([3, 2])
        with c1:
            pid = int(players.iloc[st.selectbox("Player", range(len(players)), index=idx, key=f"{key}_player",
                        format_func=lambda i: f"{players.iloc[i]['name']}" + (f" · {players.iloc[i]['country']}" if isinstance(players.iloc[i]['country'], str) else "")) ]["id"])
        row = players[players.id == pid].iloc[0]
    st.session_state["vp_player_id"] = pid
    bat_n, bowl_n = int(row["bat_n"]), int(row["bowl_n"])
    roles = [r for r, n in (("Batter", bat_n), ("Bowler", bowl_n)) if n >= 15]
    role = roles[0] if len(roles) == 1 else st.radio("Predict as", roles, index=0 if bat_n >= bowl_n else 1, horizontal=True, key=f"{key}_role")

    vlist = prof.sort_values("n_innings", ascending=False).reset_index(drop=True)
    hist_for_default = V.bat_frame(sql, pid) if role == "Batter" else V.bowl_frame(sql, pid)
    fav = int(hist_for_default.venue_id.value_counts().index[0]) if len(hist_for_default) else int(vlist.venue_id.iloc[0])
    vids = vlist.venue_id.astype(int).tolist(); vidx = vids.index(fav) if fav in vids else 0
    vid = vids[st.selectbox("Stadium (type to search)", range(len(vids)), index=vidx, key=f"{key}_venue",
                            format_func=lambda i: _venue_label(vlist.iloc[i]))]
    tour = pos = None; chasing = False
    if not compact:
        c3, c4, c5 = st.columns(3)
        last_t = hist_for_default.tournament.iloc[-1] if len(hist_for_default) else None
        cats = list(mdl["bat_cats"] if role == "Batter" else mdl["bowl_cats"])
        with c3:
            tour = st.selectbox("League context", cats, index=cats.index(last_t) if last_t in cats else 0, key=f"{key}_tour", format_func=lambda c: tour_names.get(c, c),
                                help="Leagues differ a lot in scoring. Defaults to the league he played last.")
        if role == "Batter":
            usual = int(round(hist_for_default.batting_position.tail(30).median())) if len(hist_for_default) else 4
            with c4: pos = st.slider("Batting position", 1, 11, min(max(usual, 1), 11), key=f"{key}_pos", help="Defaults to his recent usual position.")
            with c5: chasing = st.radio("Innings", ["Setting a target", "Chasing"], horizontal=True, key=f"{key}_inn") == "Chasing"

    res = V.predict_batter(sql, art, pid, vid, position=pos, chasing=chasing, tournament=tour) if role == "Batter" else V.predict_bowler(sql, art, pid, vid, tournament=tour)
    if res is None: st.info("Not enough history for this player."); return
    v = res["venue"]; bat = res["role"] == "bat"; unit = "runs" if bat else "runs / over"

    # ── headline ──
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    st.markdown(f'<div class="nb-label">{"Likely score" if bat else "Likely economy"} · {html.escape(v["name"])}</div>', unsafe_allow_html=True)
    _head = st.container(key=f"{key}_headline"); m = _head.columns(5 if bat else 4)
    m[0].metric("Expected " + ("runs" if bat else "economy"), f"{res['expected']:.1f}" if bat else f"{res['expected']:.2f}",
                f"{res['lift']:+.2f} vs his usual ground" if not bat else f"{res['lift']:+.1f} vs his usual ground", delta_color="normal" if bat else "inverse")
    m[1].metric("His baseline", f"{res['baseline']:.1f}" if bat else f"{res['baseline']:.2f}", help="What the model expects from him with no venue information, in this league and year.")
    m[2].metric("Typical range (10–90%)", f"{res['range'][0]:.0f} – {res['range'][1]:.0f}" if bat else f"{res['range'][0]:.1f} – {res['range'][1]:.1f}")
    if bat:
        m[3].metric("Chance of 30+", f"{res['p30']*100:.0f}%"); m[4].metric("Chance of 50+", f"{res['p50']*100:.0f}%")
    else:
        rec = res["record"]; m[3].metric("His record here", f"{rec['econ']:.2f}" if rec["econ"] else "never bowled here", f"{rec['overs']:.0f} overs" if rec["econ"] else None, delta_color="off")
    pct = abs(res["lift"]) / max(res["baseline"], 1e-6) * 100
    st.caption(f"**How to read this:** this ground moves his expectation by about **{res['lift']:+.1f} {unit}**, which is {pct:.1f}% of his usual "
               f"(plausible between {res['lift_lo']:+.1f} and {res['lift_hi']:+.1f}, allowing for how well we know the ground). "
               f"A single innings varies far more than that, as the range above shows. Treat the number as a tilt, not a forecast.")
    if compact:
        st.caption("Open **Predict → Venue Predictor** for the full breakdown: how this ground plays, why the number, and similar grounds."); return

    # ── how this ground plays ──
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    L, R = st.columns([6, 5])
    with L:
        st.markdown('<div class="nb-label">How this ground plays (after removing who plays there)</div>', unsafe_allow_html=True)
        fig = _fingerprint_chart(prof, vid, plot_defaults)
        if fig is not None: st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption("Teal = more than the typical ground, pink = less. Scoring is runs per 120 balls above the league average, after removing batting side, bowling side, league and year.")
    with R:
        st.markdown('<div class="nb-label">The ground</div>', unsafe_allow_html=True)
        r = prof[prof.venue_id == vid].iloc[0]; facts = [("Altitude", f"{v['elevation']:,.0f} m" if v["elevation"] is not None else None),
            ("Boundary (average)", f"{v['boundary']:.0f} m" if v["boundary"] is not None else None),
            ("Straight / square", f"{r['boundary_straight_m']:.0f} / {r['boundary_square_m']:.0f} m" if pd.notna(r.get("boundary_straight_m")) and pd.notna(r.get("boundary_square_m")) else None),
            ("Pitch type", r.get("pitch_type") if isinstance(r.get("pitch_type"), str) else None), ("Capacity", f"{int(r['capacity']):,}" if pd.notna(r.get("capacity")) else None),
            ("Innings in our data", f"{v['n']:,}"), ("Scoring vs league", f"{v['pct_vs_league']:+.1f}%")]
        st.markdown('<table class="pp-facts">' + "".join(f'<tr><td class="pp-k">{html.escape(k)}</td><td>{html.escape(str(x))}</td></tr>' for k, x in facts if x) + "</table>", unsafe_allow_html=True)
        st.progress(float(v["weight_obs"]), text=f"{v['weight_obs']*100:.0f}% from matches played here · {(1-v['weight_obs'])*100:.0f}% from altitude, boundary size and other traits")
        if v["n"] == 0: st.warning("No matches in our data here: the estimate rests on altitude, boundary size and similar traits only.")
        elif v["n"] < 30: st.info(f"Only {v['n']} innings here, so the number leans on the ground's physical traits.")

    # ── why this number ──
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="nb-label">Why this number</div>', unsafe_allow_html=True)
    parts = sorted(v["parts"].items(), key=lambda kv: -abs(kv[1]))[:4]; G = meta["G"]
    steps = [("League average team score", f"{G:.0f} runs per 120 balls", ""), ]
    steps += [(f"  physical trait: {_TRAIT_LABEL.get(k, k)}", f"{x:+.1f}", "") for k, x in parts if abs(x) >= 0.3]
    steps += [("Prior from physical traits (total)", f"{v['prior']:+.1f}", "what altitude, boundaries and similar traits alone predict")]
    if v["observed"] is not None:
        steps += [("Observed in matches here", f"{v['observed']:+.1f}", f"{v['n']} innings, adjusted for who played"), ("Blend (weight on observed)", f"{v['weight_obs']*100:.0f}%", "more innings means more trust in the observed value")]
    steps += [("Final ground effect", f"{v['final']:+.1f}  ± {v['se']:.1f}", f"{v['pct_vs_league']:+.1f}% vs league"),
              ("Scaled to this player", f"x {res['elasticity']:.2f}", "a player's runs move less than team runs (estimated from data)"),
              ("Lift on his baseline", f"{res['lift']:+.2f} {unit}", f"{res['baseline']:.1f} → {res['expected']:.1f}")]
    st.markdown('<table class="pp-facts">' + "".join(f'<tr><td class="pp-k">{html.escape(a)}</td><td>{html.escape(b)}</td><td style="opacity:.6">{html.escape(c)}</td></tr>' for a, b, c in steps) + "</table>", unsafe_allow_html=True)

    # ── his record and similar grounds ──
    st.markdown('<div class="nb-divider"></div>', unsafe_allow_html=True)
    A, B = st.columns(2)
    with A:
        st.markdown('<div class="nb-label">His record here</div>', unsafe_allow_html=True)
        rec = res["record"]
        if bat:
            if rec["inns"]: st.markdown(f"**{rec['inns']}** innings · **{rec['runs']}** runs · average **{rec['avg']:.1f}** · strike rate **{rec['sr']:.0f}** · best **{rec['best']}**")
            else: st.caption("He has not batted here in our data.")
            st.caption(f"Model expects {res['expected']:.1f}. A personal record here is shown for context only: in our backtest, past form at a specific ground did not improve predictions.")
        else:
            if rec["econ"]: st.markdown(f"**{rec['spells']}** spells · **{rec['overs']:.0f}** overs · economy **{rec['econ']:.2f}**")
            else: st.caption("He has not bowled here in our data.")
    with B:
        st.markdown('<div class="nb-label">Grounds that play like this one</div>', unsafe_allow_html=True)
        sim = V.similar_grounds(prof, vid); bv = res["by_venue"].set_index("venue_id")
        rows = []
        for r in sim.itertuples():
            if r.venue_id in bv.index:
                b = bv.loc[r.venue_id]; rows.append({"Ground": r.name, "Innings there": int(r.n_innings), "His record": (f"{int(b['inns'])} inns, avg {b['avg']:.1f}" if bat else f"{int(b['spells'])} spells, econ {b['runs']*6/b['balls']:.2f}")})
            else: rows.append({"Ground": r.name, "Innings there": int(r.n_innings), "His record": "not played"})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with st.expander("How this works, and how far to trust it"):
        bt = meta["backtest"]["results"]; pb = bt["batting"]["baseline x venue index (pooled with physics prior)"]; pw = bt["bowling"]["baseline x venue index (pooled with physics prior)"]
        st.markdown(f"""
1. **Ground effect.** Team innings (runs per 120 balls) are modelled as batting side + bowling side + league + year + **ground**. The ground term is what is left after removing who played, so a weak opposition no longer makes a ground look flat.
2. **Physical prior.** Ground effects are regressed on altitude, boundary size and shape, latitude, pitch type, floodlights and capacity (cross-validated R² on unseen grounds: **{meta['prior_cv_r2']:.2f}**, so these traits explain roughly a tenth of how grounds differ).
3. **Partial pooling.** Each ground's final effect leans on its own matches when there are many and on the physical prior when there are few. Uncertainty comes from a bootstrap.
4. **Player expectation.** His own baseline (gradient boosting on his history, no venue information) × (1 + {res['elasticity']:.2f} × ground effect ÷ league mean). The {res['elasticity']:.2f} is estimated from data.
5. **Range and probabilities.** The empirical spread of real innings by players with a similar expectation, rescaled to his level.

**Backtest (train before {meta['backtest']['split']}, test after, {meta['backtest']['n_test_innings']:,} innings).** The ground adjustment improved bowler error significantly (RMSE {pw['all test']['rmse']:.3f} vs {bt['bowling']['baseline: bowler only (no venue)']['all test']['rmse']:.3f}) and batter RMSE very slightly ({pb['all test']['rmse']:.3f} vs {bt['batting']['baseline: player only (no venue)']['all test']['rmse']:.3f}); the typical-innings error (MAE) was not better for batters. The ground is a real but small effect, and one innings is mostly luck.""")


def render_venue_compare(sql, plot_defaults=None, seed_player_id=None, key="vc"):
    """Compare up to 8 players at one stadium: baseline, expected when setting and when chasing, lift, range and chances, plus economy for bowlers."""
    art = _art(sql, "v1")
    if art is None: st.warning("The venue model has not been built yet. Run `python scripts/build_venue_model.py --build`."); return
    prof, meta, mdl = art; players = _players(sql, "v1"); tour_names = _tour_names(sql)
    ids = players["id"].astype(int).tolist(); name_of = {int(r.id): r.name + (f" · {r.country}" if isinstance(r.country, str) else "") for r in players.itertuples()}
    seed = int(seed_player_id) if seed_player_id is not None and int(seed_player_id) in ids else ids[0]
    top = players[players.id != seed].sort_values("bat_n", ascending=False).head(3)["id"].astype(int).tolist()
    st.markdown('<div class="nb-label">Compare players at one stadium (max 8)</div>', unsafe_allow_html=True)
    chosen = st.multiselect("Players", ids, default=[seed] + top, max_selections=8, format_func=lambda i: name_of[i], key=f"{key}_players")
    vlist = prof.sort_values("n_innings", ascending=False).reset_index(drop=True); vids = vlist.venue_id.astype(int).tolist()
    hist0 = V.bat_frame(sql, seed); fav = int(hist0.venue_id.value_counts().index[0]) if len(hist0) else vids[0]
    c1, c2 = st.columns([3, 2])
    with c1:
        vid = vids[st.selectbox("Stadium (type to search)", range(len(vids)), index=vids.index(fav) if fav in vids else 0, key=f"{key}_venue", format_func=lambda i: _venue_label(vlist.iloc[i]))]
    with c2:
        cats = ["Each player's latest league"] + list(mdl["bat_cats"])
        tour = st.selectbox("League context", cats, key=f"{key}_tour", format_func=lambda c: c if c == cats[0] else tour_names.get(c, c))
    if not (chosen and st.button("Compare at stadium", key=f"{key}_go")): return
    rows = []
    with st.spinner("Predicting…"):
        for pid in chosen:
            info = players[players.id == pid].iloc[0]; rec = {"Player": info["name"], "Country": info["country"] if isinstance(info["country"], str) else "—"}
            tour_arg = None if tour == cats[0] else tour
            if int(info["bat_n"]) >= 15:
                h = V.bat_frame(sql, pid); a = V.predict_batter(sql, art, pid, vid, chasing=False, tournament=tour_arg, hist=h); b = V.predict_batter(sql, art, pid, vid, chasing=True, tournament=tour_arg, hist=h)
                rec.update({"Pos": a["pos"], "Baseline runs": a["baseline"], "Setting": a["expected"], "Chasing": b["expected"], "Lift": a["lift"],
                            "Range 10-90%": f"{a['range'][0]:.0f}-{a['range'][1]:.0f}", "30+": a["p30"] * 100, "50+": a["p50"] * 100})
            if int(info["bowl_n"]) >= 15:
                w = V.predict_bowler(sql, art, pid, vid, tournament=tour_arg); rec.update({"Baseline econ": w["baseline"], "Econ here": w["expected"], "Econ lift": w["lift"]})
            rows.append(rec)
    df = pd.DataFrame(rows)
    if "Setting" in df: df = df.sort_values("Setting", ascending=False, na_position="last")
    st.markdown(f'<div class="nb-label">{html.escape(prof[prof.venue_id == vid].iloc[0]["name"])}</div>', unsafe_allow_html=True)
    cfg = {"Baseline runs": st.column_config.NumberColumn(format="%.1f"), "Setting": st.column_config.NumberColumn("Setting target", format="%.1f"),
           "Chasing": st.column_config.NumberColumn(format="%.1f"), "Lift": st.column_config.NumberColumn("Venue lift", format="%+.1f"),
           "30+": st.column_config.NumberColumn("Chance 30+", format="%.0f%%"), "50+": st.column_config.NumberColumn("Chance 50+", format="%.0f%%"),
           "Baseline econ": st.column_config.NumberColumn(format="%.2f"), "Econ here": st.column_config.NumberColumn(format="%.2f"), "Econ lift": st.column_config.NumberColumn(format="%+.2f")}
    st.dataframe(df.reset_index(drop=True), hide_index=True, width="stretch", column_config=cfg)
    if "Setting" in df and df["Setting"].notna().any():
        d = df.dropna(subset=["Setting"]); fig = go.Figure()
        for nm, col, colr in (("His usual (baseline)", "Baseline runs", "#B8B8B8"), ("Setting a target", "Setting", "#3A86FF"), ("Chasing", "Chasing", "#FF6B9D")):
            fig.add_trace(go.Bar(name=nm, x=d["Player"], y=d[col], marker=dict(color=colr, line=dict(color="#0D0D0D", width=2))))
        fig.update_layout(barmode="group", title="Expected runs at this stadium", yaxis_title="runs")
        st.plotly_chart(plot_defaults(fig, 340) if plot_defaults else fig, width="stretch", config={"displayModeBar": False})
    st.caption("Venue lifts are small (about 1 to 2 runs for a typical batter). Differences between players are mostly the players, not the ground. Ranges and chances are for a single innings.")
