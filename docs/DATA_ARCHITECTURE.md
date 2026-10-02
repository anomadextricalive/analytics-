# Cricket data architecture (decided 2026-10-03)

## Layers

```
raw (immutable)            truth (SQLite)                derived (MongoDB)               readers
Cricsheet JSON      ─┐
ESPN ball-by-ball   ─┼─►  data/cricket.db  ──build──►   cricket_serving  (documents) ──► web app (fixed endpoints)
Statsguru raw .gz   ─┤     WAL mode, one file           cricket_analytics (table mirror) ► hosted Streamlit
Cricbuzz (gap-fill) ─┘
```

1. **Raw is never edited.** Statsguru pages stay in `statsguru/raw/*.html.gz`, crawl state in `statsguru/bulk.db`.
2. **SQLite is the only source of truth.** Every fix lands there first. It must stay in WAL mode.
3. **Mongo is derived and rebuildable.** Nothing writes to it except the build scripts. Each collection is built into
   `<name>__staging`, count-checked, then swapped in; the previous live copy is kept as `<name>__prev` (one generation).
4. **Two Mongo databases, two jobs.**
   - `cricket_serving`: document read model for the public web app. Documents are shaped for one page each.
     No model-written SQL or free-form queries on a public site; the app calls fixed, parameterised endpoints.
   - `cricket_analytics`: 1:1 mirror of SQLite tables for the hosted Streamlit app. Frozen to the existing tables,
     no new features go here.
5. **Storage stays Mongo, not Neon,** for the web version. Neon would need the paid plan for 2.8M deliveries; the
   serving DB is ~440 MB of pre-joined documents.

## Identity

- **ESPN Cricinfo id is the person key** across sources. Cricsheet's register gives it for 8,091 of our 8,092 players.
- Internal `players.id` stays as the key for anything derived from match data. Bridge: `player_espn_map`
  (`status = 'matched'`).
- Players with no match data here (about 5,900 of the 13,908 in Statsguru) exist only as `people` docs.
- 2,973 Cricbuzz-created players have no ESPN id yet; they appear in `players` but not `people`.

## Two kinds of career numbers, kept apart

| | Source | Covers | Where |
|---|---|---|---|
| Ours | summed from `player_innings` / `player_bowling_innings` | only matches we hold, per tournament | `players.career_bat`, `career_bowl` |
| Reference | Statsguru (ESPN Cricinfo) headline line | every T20 the player ever played | `espn_career` table, `people`, `players.career_espn` |

They will differ until match coverage reaches ESPN's (about 6,500 T20s missing). UI should label them and never mix.

## Schema

### SQLite `espn_career` (exists)
PK `(espn_id, fmt, stat_type)`. `fmt` = `t20` | `t20i`, `stat_type` = `batting` | `bowling`, `status` = `ok` | `none` | `error`.
Batting: `mat inns nos runs bf hs ave sr hundreds fifties ducks fours sixes`. Bowling: `mat inns balls overs runs wkts bbi ave econ sr four_w five_w mdns`.
`source` and `raw_json` keep provenance; `fetched_at` is the crawl time.

### Mongo `cricket_serving.people` (new)
```
_id: "485562"            // ESPN id as string
espn_id: 485562
name: "G Dukes"
country: "England"       // only when linked to players (Statsguru has no country column)
player_id: 1234          // players._id when we hold their matches
t20:  { bat:  {span, mat, inns, nos, runs, hs, ave, bf, sr, hundreds, fifties, ducks, fours, sixes},
        bowl: {span, mat, inns, balls, runs, wkts, bbi, ave, econ, sr, four_w, five_w, mdns} }
t20i: { bat, bowl }      // same shape
```
Indexes: `player_id`, `country`.

### Mongo `cricket_serving.players` (extended)
Adds `espn_id` and `career_espn: { t20: {bat, bowl}, t20i: {bat, bowl} }` beside the existing per-tournament blocks.
Index on `espn_id`.

### Unchanged
`matches`, `venues`, `tournaments`, `leaderboards`, `innings_balls` as in `scripts/build_mongo_serving.py`.

## Not decided here
- `match_source` table (coverage level per match: ball-by-ball / scorecard-only). Needs the scorecard-only decision first.
- Extra ball fields (shot, pitch, wagon, win probability). Would add columns to `deliveries`; ESPN top-tier matches only.
- Women's T20 stays out of scope.

## Refresh procedure
1. `crawl` Statsguru (1 worker, never bursts) → `scripts/merge_statsguru_bulk.py --check-only [--overrides FILE]` → merge.
   Players who played mid-crawl show up as cross-pass disagreements; refetch them with `CricinfoClient.player_career_stats`
   and pass the result via `--overrides`.
2. Back up `data/cricket.db` to `~/etpl2026/backups/` before any merge.
3. `python scripts/build_mongo_serving.py` (serving) and `python scripts/migrate_to_mongo.py` (mirror).
