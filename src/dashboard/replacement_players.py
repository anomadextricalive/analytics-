"""ADT10 replacement-candidate scouting and squad style-balance views."""

import re
from pathlib import Path

import pandas as pd
import streamlit as st


def _normalise_name(value):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z ]", " ", str(value or "").lower().replace("-", " "))).strip()


def _query(sql, query_fn):
    try:
        return query_fn(sql)
    except Exception:
        return pd.DataFrame()


def _format_career_table(raw, discipline):
    if raw.empty or not {"player_id", "fmt"}.issubset(raw.columns):
        return pd.DataFrame()
    metrics = [column for column in raw.columns if column not in {"player_id", "fmt"}]
    wide = raw.pivot(index="player_id", columns="fmt", values=metrics)
    wide.columns = [f"{fmt}_{discipline}_{metric}" for metric, fmt in wide.columns]
    return wide.reset_index()


def _load_candidates(path):
    candidates = pd.read_csv(path, dtype=str, keep_default_na=False)
    candidates["Player"] = (candidates["First Name"].str.strip() + " "
                            + candidates["Last Name"].str.strip()).str.strip()
    candidates["Age"] = pd.to_numeric(candidates["Age"], errors="coerce")
    candidates["Database Name"] = ""
    candidates["Database ID"] = pd.NA
    candidates["Identity Match"] = "No exact database match"
    return candidates


def _add_database_stats(candidates, query_fn):
    aliases = _query(
        "SELECT player_id, alias_norm FROM player_aliases", query_fn
    )
    if aliases.empty or not {"player_id", "alias_norm"}.issubset(aliases.columns):
        return candidates

    alias_groups = aliases.groupby("alias_norm")["player_id"].nunique()
    unique_aliases = aliases[aliases["alias_norm"].isin(alias_groups[alias_groups == 1].index)]
    id_by_alias = unique_aliases.drop_duplicates("alias_norm").set_index("alias_norm")["player_id"]
    all_alias_groups = aliases.groupby("alias_norm")["player_id"].nunique()

    candidates = candidates.copy()
    candidates["_normalized_name"] = candidates["Player"].map(_normalise_name)
    candidates["Database ID"] = candidates["_normalized_name"].map(id_by_alias)
    ambiguous = candidates["_normalized_name"].map(all_alias_groups).fillna(0).gt(1)
    candidates.loc[ambiguous, "Identity Match"] = "Ambiguous name — stats withheld"
    matched = candidates["Database ID"].notna()
    candidates.loc[matched, "Identity Match"] = "Exact alias match"

    player_names = _query(
        "SELECT id AS player_id, COALESCE(NULLIF(full_name, ''), cricsheet_key) AS database_name "
        "FROM players", query_fn
    )
    if not player_names.empty:
        name_by_id = player_names.drop_duplicates("player_id").set_index("player_id")["database_name"]
        candidates.loc[matched, "Database Name"] = candidates.loc[matched, "Database ID"].map(name_by_id)

    stat_queries = {
        "Batting": """SELECT player_id, innings AS bat_innings, not_outs, runs AS bat_runs,
            balls AS bat_balls, hs AS high_score, thirties, fifties, hundreds, ducks,
            fours, sixes, median_score, times_opened, top_scored, average AS bat_average,
            strike_rate AS bat_strike_rate, pp_sr, mid_sr, death_sr, adj_average,
            adj_strike_rate FROM player_career_bat WHERE tournament = 'ALL'""",
        "Bowling": """SELECT player_id, innings AS bowl_innings, balls AS bowl_balls,
            runs AS bowl_runs, wickets, dot_balls, economy, average AS bowl_average,
            strike_rate AS bowl_strike_rate, dot_pct, pp_economy, mid_economy,
            death_economy, adj_economy FROM player_career_bowl WHERE tournament = 'ALL'""",
        "Ratings": """SELECT player_id, bat_rating, bowl_rating, overall_rating, opener_score,
            finisher_score, anchor_score, chase_score, pp_bat_score, death_bat_score,
            pp_bowl_score, mid_bowl_score, death_bowl_score FROM player_ratings
            WHERE tournament = 'ALL'""",
        "Form": """SELECT player_id, avg_5, avg_10, avg_20, sr_5, sr_10, sr_20,
            career_avg, career_sr, cv, breakout_flag, breakout_delta, innings_total
            FROM player_form""",
    }

    for label, statement in stat_queries.items():
        stats = _query(statement, query_fn)
        if stats.empty or "player_id" not in stats.columns:
            continue
        candidates = candidates.merge(stats, left_on="Database ID", right_on="player_id", how="left")
        candidates.drop(columns="player_id", inplace=True, errors="ignore")

    international_fields = {
        "batting": "mat, inns, runs, ave, sr, bf, hs, hundreds, fifties, ducks, fours, sixes, no, span",
        "bowling": "mat, inns, wkts, ave, econ, sr, bbi, overs, mdns, four_w, five_w, span",
    }
    for discipline, fields in international_fields.items():
        raw = _query(
            f"SELECT player_id, fmt, {fields} FROM player_career_intl WHERE stat_type = '{discipline}'",
            query_fn,
        )
        formatted = _format_career_table(raw, discipline)
        if not formatted.empty:
            candidates = candidates.merge(formatted, left_on="Database ID", right_on="player_id", how="left")
            candidates.drop(columns="player_id", inplace=True, errors="ignore")

    candidates.drop(columns="_normalized_name", inplace=True, errors="ignore")
    return candidates


def _bowling_group(style):
    value = str(style or "").lower()
    if "not applicable" in value or not value:
        return ""
    if any(word in value for word in ("fast", "medium", "seam")):
        return "Left-arm pace" if "left-arm" in value else "Right-arm pace"
    if any(word in value for word in ("break", "spin", "orthodox", "chinaman")):
        if "left-arm" in value or "slow left" in value:
            return "Left-arm spin"
        if "leg" in value or "wrist" in value or "chinaman" in value:
            return "Right-arm wrist-spin"
        return "Right-arm finger-spin"
    return ""


def _player_label(row):
    age = f" · {int(row['Age'])}" if pd.notna(row["Age"]) else ""
    return f"{row['Player']}{age} · {row['Draft Category']}"


def _render_metric_group(row, columns, labels):
    available = [(column, label) for column, label in zip(columns, labels)
                 if column in row.index and pd.notna(row[column])]
    if not available:
        st.info("No linked stats are available for this section.")
        return
    cards = st.columns(min(4, len(available)))
    for index, (column, label) in enumerate(available):
        value = row[column]
        if isinstance(value, float):
            value = f"{value:.1f}"
        cards[index % len(cards)].metric(label, str(value))
    details = pd.DataFrame([{"Metric": label, "Value": row[column]}
                            for column, label in available])
    st.dataframe(details, hide_index=True, use_container_width=True)


def render_replacement_scout(data_path, query_fn):
    st.markdown("""
    <div class="nb-page-header">
      <h2>ADT10 Replacement Scout</h2>
      <p>Replacement candidates · T20 evidence · role and style balance</p>
    </div>""", unsafe_allow_html=True)

    path = Path(data_path)
    if not path.exists():
        st.error("Replacement-player data is missing. Rebuild it with scripts/import_adt10_replacement_players.py.")
        return

    candidates = _add_database_stats(_load_candidates(path), query_fn)
    left, right = st.columns([1, 1])
    with left:
        st.metric("Candidates", f"{len(candidates):,}")
    with right:
        st.metric("Exact analytics matches", f"{candidates['Database ID'].notna().sum():,}")
    st.caption("Linked performance comes from the local T20 analytics database. An unmatched player has no exact alias; ambiguous names are deliberately left unlinked.")

    category_values = sorted({part.strip() for value in candidates["Draft Category"]
                              for part in value.split(",") if part.strip()})
    role_values = sorted(candidates["Player Role"].replace("", pd.NA).dropna().unique())
    availability_values = sorted(candidates["Availability"].replace("", pd.NA).dropna().unique())
    f1, f2, f3, f4 = st.columns([1.5, 1, 1, 1])
    with f1:
        search = st.text_input("Search player or country", key="replacement_search")
    with f2:
        categories = st.multiselect("Draft category", category_values, key="replacement_categories")
    with f3:
        roles = st.multiselect("Role", role_values, key="replacement_roles")
    with f4:
        availability = st.multiselect("Availability", availability_values, key="replacement_availability")

    filtered = candidates.copy()
    if search.strip():
        search_blob = (filtered["Player"] + " " + filtered["Cricket Board"] + " "
                       + filtered["Nationality"]).str.lower()
        filtered = filtered[search_blob.str.contains(search.strip().lower(), regex=False)]
    if categories:
        filtered = filtered[filtered["Draft Category"].apply(
            lambda value: any(category in [part.strip() for part in value.split(",")]
                              for category in categories))]
    if roles:
        filtered = filtered[filtered["Player Role"].isin(roles)]
    if availability:
        filtered = filtered[filtered["Availability"].isin(availability)]

    st.markdown("#### Candidate pool")
    visible = [
        "Player", "Age", "Cricket Board", "Draft Category", "Player Role",
        "Batting Style", "Bowling Style", "T20 Internationals",
        "T20 Domestic Matches", "Availability", "Identity Match", "bat_runs",
        "bat_average", "bat_strike_rate", "wickets", "economy", "overall_rating",
        "avg_5", "avg_10", "sr_10",
    ]
    visible = [column for column in visible if column in filtered.columns]
    st.dataframe(filtered[visible], hide_index=True, use_container_width=True, height=380)
    st.caption(f"Showing {len(filtered):,} candidates. Blank analytics fields mean no linked database value.")

    st.markdown("#### Shortlist")
    shortlist_options = sorted(filtered["Player"].drop_duplicates().tolist())
    shortlist = st.multiselect("Players to keep on your shortlist", shortlist_options,
                               key="replacement_shortlist")
    if shortlist:
        export = filtered[filtered["Player"].isin(shortlist)].drop(
            columns=[column for column in filtered if column.startswith("_")], errors="ignore")
        st.download_button("Download shortlist CSV", export.to_csv(index=False).encode("utf-8-sig"),
                           "adt10-replacement-shortlist.csv", "text/csv")

    st.markdown("#### Player profile")
    profile_names = filtered["Player"].drop_duplicates().tolist()
    if profile_names:
        selected_name = st.selectbox("Select a candidate", profile_names, key="replacement_profile")
        profile = candidates[candidates["Player"] == selected_name].iloc[0]
        st.markdown(f"**{_player_label(profile)}** · {profile['Cricket Board']} · {profile['Player Role']}")
        profile_url = re.search(r"https?://\S+", str(profile["Cricinfo Profile Link"]))
        if profile_url:
            st.markdown(f"[Open player profile]({profile_url.group(0)})")
        t20_tabs = st.tabs(["T20 batting", "T20 bowling", "International career", "Ratings & form", "Roster details"])
        with t20_tabs[0]:
            _render_metric_group(profile,
                ["bat_innings", "bat_runs", "bat_average", "bat_strike_rate", "high_score",
                 "fours", "sixes", "fifties", "hundreds", "ducks", "pp_sr", "mid_sr",
                 "death_sr", "adj_average", "adj_strike_rate"],
                ["Innings", "Runs", "Average", "Strike rate", "High score", "4s", "6s",
                 "50s", "100s", "Ducks", "Powerplay SR", "Middle SR", "Death SR",
                 "Venue-adjusted average", "Venue-adjusted SR"])
        with t20_tabs[1]:
            _render_metric_group(profile,
                ["bowl_innings", "bowl_balls", "wickets", "economy", "bowl_average",
                 "bowl_strike_rate", "dot_pct", "pp_economy", "mid_economy", "death_economy",
                 "adj_economy"],
                ["Innings", "Balls", "Wickets", "Economy", "Average", "Strike rate",
                 "Dot ball %", "Powerplay econ", "Middle econ", "Death econ", "Adjusted econ"])
        with t20_tabs[2]:
            _render_metric_group(profile,
                ["t20i_bat_mat", "t20i_bat_inns", "t20i_bat_runs", "t20i_bat_ave",
                 "t20i_bat_sr", "t20i_bat_hs", "t20i_bat_fifties", "t20i_bat_hundreds",
                 "t20i_bowl_wkts", "t20i_bowl_ave", "t20i_bowl_econ", "t20i_bowl_sr",
                 "t20i_bowl_bbi", "t20i_bowl_five_w"],
                ["T20I bat matches", "Bat innings", "Runs", "Average", "Strike rate", "High score",
                 "50s", "100s", "Wickets", "Bowl average", "Economy", "Bowl strike rate",
                 "Best bowling", "5-wicket hauls"])
            international_formats = sorted({
                column.split("_")[0] for column in profile.index
                if re.match(r"^(t20i|odi|test|fc|lista|t20)_", column)
            })
            for fmt in international_formats:
                batting = {column.removeprefix(f"{fmt}_bat_"): profile[column]
                           for column in profile.index if column.startswith(f"{fmt}_bat_")
                           and pd.notna(profile[column])}
                bowling = {column.removeprefix(f"{fmt}_bowl_"): profile[column]
                           for column in profile.index if column.startswith(f"{fmt}_bowl_")
                           and pd.notna(profile[column])}
                if batting:
                    st.markdown(f"**{fmt.upper()} batting**")
                    st.dataframe(pd.DataFrame([batting]), hide_index=True, use_container_width=True)
                if bowling:
                    st.markdown(f"**{fmt.upper()} bowling**")
                    st.dataframe(pd.DataFrame([bowling]), hide_index=True, use_container_width=True)
        with t20_tabs[3]:
            _render_metric_group(profile,
                ["overall_rating", "bat_rating", "bowl_rating", "opener_score", "finisher_score",
                 "anchor_score", "chase_score", "pp_bat_score", "death_bat_score", "pp_bowl_score",
                 "mid_bowl_score", "death_bowl_score", "avg_5", "avg_10", "avg_20", "sr_5",
                 "sr_10", "sr_20", "breakout_delta", "innings_total"],
                ["Overall", "Batting", "Bowling", "Opener", "Finisher", "Anchor", "Chase",
                 "PP batting", "Death batting", "PP bowling", "Middle bowling", "Death bowling",
                 "Last 5 avg", "Last 10 avg", "Last 20 avg", "Last 5 SR", "Last 10 SR",
                 "Last 20 SR", "Form vs career", "Form innings"])
            st.caption("Ratings and recent-form values are shown only when calculated by the analytics database.")
        with t20_tabs[4]:
            detail_columns = ["Player", "Age", "Cricket Board", "Draft Category", "Player Role",
                              "Batting Style", "Bowling Style", "T20 Internationals",
                              "T20 Domestic Matches", "Availability", "Availability Notes",
                              "Nationality", "Identity Match", "Database Name"]
            st.dataframe(pd.DataFrame([profile[detail_columns]]).T.rename(columns={profile.name: "Value"}),
                         use_container_width=True)

    st.markdown("#### XI style balance")
    st.caption("A checklist of complementary options, not a player-quality ranking. Use the cricket context and team rules when choosing the final XI.")
    team_options = sorted(candidates["Player"].drop_duplicates().tolist())
    xi = st.multiselect("Choose up to 11 candidates", team_options, max_selections=11,
                        key="replacement_xi")
    if xi:
        selected = candidates[candidates["Player"].isin(xi)].copy()
        selected["Bowling family"] = selected["Bowling Style"].map(_bowling_group)
        left_bat = selected["Batting Style"].str.contains("left", case=False, na=False).any()
        right_bat = selected["Batting Style"].str.contains("right", case=False, na=False).any()
        keeper = selected["Player Role"].str.contains("wk|wicket", case=False, regex=True, na=False).any()
        left_arm = selected["Bowling Style"].str.contains("left-arm|slow left", case=False, regex=True, na=False).any()
        bowling_families = sorted(set(selected["Bowling family"]) - {""})
        a, b, c, d = st.columns(4)
        a.metric("Selected", f"{len(selected)} / 11")
        b.metric("Batting hands", f"{int(left_bat) + int(right_bat)} / 2")
        c.metric("Bowling families", len(bowling_families))
        d.metric("Listed keeper", "Yes" if keeper else "No")
        gaps = []
        if not keeper:
            gaps.append("No listed wicketkeeper")
        if not left_bat:
            gaps.append("No left-handed batting option")
        if not left_arm:
            gaps.append("No left-arm bowling option")
        if len(bowling_families) < 3:
            gaps.append("Fewer than three distinct bowling families")
        if gaps:
            st.warning("Coverage to consider: " + " · ".join(gaps))
        else:
            st.success("The selected group covers the listed role and style checklist.")
        st.dataframe(selected[["Player", "Player Role", "Batting Style", "Bowling Style",
                               "Identity Match", "overall_rating", "avg_10", "sr_10"]],
                     hide_index=True, use_container_width=True)
