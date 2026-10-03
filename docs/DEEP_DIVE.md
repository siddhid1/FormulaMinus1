# F1 Replay — Developer Internals Deep Dive

How the application turns raw FastF1 telemetry into a 60 fps animated race
replay: the data pipeline, the algorithms (with the math), the rendering
architecture, performance characteristics, and a postmortem of the real bugs
found while building it.

> All references are `file:line` against the current `main` branch.

---

## 1. Big picture

```
                    ┌────────────────────────── one-time per session ──────────────────────────┐
 FastF1 API ──► loader.py ──► raw cache ──► preprocess.py ──► ReplayDataset ──► *.pkl (~1 s load)
  (HTTP)        (sqlite +     (cache/fastf1)   (~8 s build)    (in-memory)
                 session                                                                │
                 files)                                                                 ▼
                    └────────────────────────────────────────────────────────────  cached forever┘

 ┌──────────────── per frame (60 fps) ────────────────┐
 │ Playback.clock ──► frame() ──► FrameState           │
 │                                ├─► track_render.py   │  track map + cars + SC
 │                                ├─► hud.py            │  top bar, leaderboard, transport
 │                                └─► panels.py         │  telemetry / lap chart / stints
 │ App.handle_events() ──► input state (selection,      │
 │   drag, scrub, toggles)                              │
 └──────────────────────────────────────────────────────┘
```

### The time model

Everything lives on one axis: **session time** in seconds, the same domain
FastF1's `SessionTime` column uses (0 = session start, which for a race is
*before* lights out).

| Symbol | Meaning | Where |
| --- | --- | --- |
| `started` | `session_status == "Started"` (green flag) | `preprocess.py:218` |
| `t0 = started − 20` | replay start: 20 s of formation anticipation | `preprocess.py:331` |
| `t1` | min(last sample, `finished + 60 s`) — a little post-flag coast | `preprocess.py:331-336` |
| `elapsed` | `t − (t0 + 20)` = the on-screen race clock (0 at lights out) | `playback.py:185` |
| `finished_at` | chequered flag time; after it, official classification wins | `preprocess.py:360` |

Replay seconds ≠ race-clock seconds only during the first 20 s; after that
they advance identically, scaled by the playback speed factor.

---

## 2. Acquisition & caching (`loader.py`)

- `enable_cache()` (`loader.py:18`) points FastF1's cache at the
  project-local `cache/fastf1/` — **never** the user's global FastF1 cache,
  so the project is self-contained and `--clear-cache` is total.
- `load_session()` (`loader.py:24`) runs two-phase load:
  `sess.load(laps=True, messages=True)` first (needed for track/status/laps),
  then `sess.load(telemetry=True)` (car + position channels). Downloads are
  HTTP-cached by FastF1 in `cache/fastf1/fastf1_http_cache.sqlite`.
- `cache_key()` (`loader.py:39`) = `{year}_{gp-slug}_{session}` — this is
  both the preprocessed pickle's filename and the selector for
  `--clear-cache <key>`.
- Two cache layers exist because they have different lifecycles:
  - **raw** — immutable source data; deleting it forces a re-download.
  - **preprocessed pickle** — derived; `--rebuild` regenerates it from raw
    without any network I/O.

`preprocess.load_dataset()` (`preprocess.py:37`) is the gate: pickle hit →
load (~1 s); miss → `load_session` + `build_dataset` (~8 s) + pickle write.

---

## 3. Preprocessing (`preprocess.py`) — the heart

### 3.1 Track construction

`build_track()` (`preprocess.py:73`) takes **one lap's** position samples
(fastest lap by default) and:

1. converts FastF1's raw **decimetres → metres** (`/10.0`);
2. de-duplicates points closer than 0.3 m (`_dedupe`, `preprocess.py:66`) —
   GPS samples at low speed would otherwise pile up;
3. closes the loop (`pts[0]` appended if not already within 0.3 m);
4. builds the cumulative arc-length array `cum` by segment-wise `cumsum`.

`_build_track_robust()` (`preprocess.py:375`) adds fallbacks: if the fastest
lap's data is broken, try other laps, then *any* lap, and sanity-check the
result length is in (1 km, 30 km). Vegas 2023: **6152.8 m, ~365 vertices**.

### 3.2 Projection: `(x, y) → (s, perp)` — `Projector`

`preprocess.py:87`. For every telemetry sample:

1. **cKDTree nearest vertex** — O(log N) per query, batched for the whole
   driver (`tree.query(xy)`, `preprocess.py:101`).
2. **Segment refinement twice**: project onto the forward segment
   `verts[idx] → verts[idx+1]` *and* the backward segment
   `verts[idx−1] → verts[idx]`. The refinement is the standard clamped
   dot-product parametrization (`_proj`, `preprocess.py:116-127`):

   ```
   f  = clamp( ((p − a) · (b − a)) / |b − a|² , 0, 1 )
   closest = a + f·(b − a)
   s   = s_a + f·|segment|
   perp = |p − closest|
   ```

3. Keep whichever of the two candidates has the **smaller perpendicular**
   (`preprocess.py:111-113`), then wrap `s` into `[0, L)`.

Why both directions? The nearest *vertex* isn't always the nearest
*segment's start* — near segment ends, the true closest point lies on the
neighbouring segment, and a single-direction refinement would snap to the
wrong one. Taking `min(perp)` over both neighbours fixes this at trivial
cost.

### 3.3 Race progress: the lap-counting bug and its fix

**Problem.** Ranking cars by "laps completed + s" requires unwrapping `s`
into a monotonically increasing race distance. Naively counting `s → 0`
wraps as completed laps fails in two real ways:

- **Pit-lane seam crossings.** The pit entry/exit path crosses the
  start/finish seam *off* the racing line; a car entering the pits near S/F
  can bounce `s` across zero **twice per pit stop**, faking extra laps.
- **Start/finish jitter.** Samples within a few metres of `s = 0` can
  oscillate across the seam due to projection noise.

Either effect hands a driver phantom laps and wrecks the leaderboard.

**Fix** — `_integrate_progress()` (`preprocess.py:140`): don't count
crossings, *integrate* per-sample deltas and fold outliers:

```
ds[i] = s[i+1] − s[i]
if ds < −L/2 :  ds += L     # forward seam crossing  (0.9L jump → real step)
if ds >  +L/2 :  ds −= L     # backward projection glitch
race_dist = s[0] + cumsum(ds)
```

Because consecutive samples are ~1 Hz on a ≤ 330 km/h car (≤ ~100 m apart),
any jump larger than half a lap (Vegas: >3 km) is by definition an artefact
and is folded into the sign of real motion. Pit seam crossings become a
normal small forward step; noise becomes a wash. Grid-position offsets are
deliberately **not** subtracted (they'd double-count before lights out);
pre-race samples simply sit near `s = grid slot`.

`race_dist` (stored as float32) is the single ranking key used every frame.

### 3.4 Per-driver channel assembly

`build_dataset()` (`preprocess.py:231-317`) loops over `results`:

- `car.merge_channels(pos, frequency="original")` merges speed/gear/DRS with
  X/Y on their common timestamps (`preprocess.py:237`).
- Clean-up: drop NaN positions, sort by `Date`, drop duplicate timestamps
  (keep last), require ≥10 samples.
- Extract `t` (SessionTime, s), `xy` (metres), `speed`, `gear`, `drs`;
  project once → `s`, `perp`; integrate once → `race_dist`.
- Laps: `LapStartTime / LapNumber / LapTime / Compound` sorted by lap;
  `_stints_from_laps()` (`preprocess.py:160`) collapses compound runs into
  `Stint(lap_from, lap_to, compound)`.
- **Retirement time** — `_retirement_time()` (`preprocess.py:417`): a
  non-`Finished` driver is marked OUT at its **last movement**: the final
  timestamp where consecutive samples displaced >3 m. A car parked in the
  pits >60 s before its telemetry ends uses that last-movement time + 1 s;
  otherwise `t[-1] + 1`. `Finished` → `out_time = None`.
- Everything stored **float32** — halves the pickle size with no visible
  precision loss at track scale (sub-millimetre at 5-digit metres? metres
  stay exact to ~1e-7 relative; irrelevant error).

### 3.5 Pit lane geometry

Evidence collection (`preprocess.py:254-276`):

- A sample is "pit lane" if `perp > 10 m` **and** within 1000 m of the
  S/F seam (`PIT_LATERAL_THRESHOLD` / `PIT_SEAM_WINDOW`, `models.py:9-10`),
  or FastF1's own `Status == "OffTrack"` near the seam. The seam window
  exists because on *some* circuits the racing line itself is metres off the
  fitted polyline in corners; only near S/F is "off the line" reliable.
- **Entry** = transition on→off racing line in the *far* half of the loop
  (`s > L/2`); **exit** = off→on transition in the *near* half
  (`preprocess.py:272-275`). Directional split disambiguates entry from
  exit at circuits where both sit near the seam. Candidates are combined by
  **median** (`preprocess.py:323-324`), with fallbacks `L−200` / `+200`.
- `_pit_lane_polyline()` (`preprocess.py:473`) averages actual pit-lane GPS
  samples into a drawable reference: seam-centre coordinate
  (`s > L/2 → s − L`), 10 m bins, mean of bins with ≥3 samples, ≥4 bins
  required. This polyline is what the Safety Car "drives out of".

### 3.6 DRS zone detection

Heuristic, fully offline (`_extract_drs_zones`, `preprocess.py:435`):

1. **Evidence**: every on-track sample (`perp < 8 m`) with `DRS ≥ 8` (open
   or available) contributes its `s` (`preprocess.py:261-263`).
2. **Histogram**: 50 m bins over `[0, L)`.
3. **Relative threshold**: `max(3, ⌈0.04 · peak⌉)` — a bin must hold ≥4% of
   the busiest bin's count. The *relative* form was a bug fix (§10): an
   absolute threshold once merged everything into one 5 km "zone".
4. **Runs**: contiguous active bins → candidate zones; drop runs <80 m.
5. **Merge** gaps ≤200 m, including across the seam
   (`preprocess.py:464-467`, may produce `b > L`, handled by the renderer).

Vegas 2023 result: `(600, 1450)` and `(4200, 5000)` — the two long
straights. ✔

### 3.7 Session-wide extraction

- **SC windows** (`preprocess.py:209-214`): `track_status` rows with
  `Status == "4"`, each paired with the next status event as its end —
  Vegas has two: `[1165, 1675]` and `[3682, 4035]` session-seconds.
- **Status events** feed the on-screen pill (GREEN / YELLOW / SAFETY CAR /
  VSC / RED FLAG) via `playback._current_status` (`playback.py:206`).
- **Official order**: `_class_sort_key` (`preprocess.py:403`) sorts by the
  results `Position` column (999 for unclassified) — used only *after*
  `finished_at`.
- **Corners** (`preprocess.py:499`): FastF1's circuit-info markers come in
  inconsistent unit systems across eras. The loader scores three candidates
  (raw, dm→m, m→dm) by **median KD distance to the track polyline** and
  accepts the best only if <60 m. (Vegas 2023: 17 corners, dm→m wins.)

---

## 4. Playback engine (`playback.py`)

- **Clock** (`Playback`, `playback.py:83`): `t` advances by
  `dt × speed` where `speed ∈ {0.5, 1, 2, 4}` (`SPEEDS`, line 11); clamps
  at `t1` and auto-pauses; `seek`/`seek_fraction` clamp to `[t0, t1]`;
  `restart` returns to `t0`.
- **Sampling a driver** (`playback.py:136-141`): positions/speed use
  `_sample_scalar` (linear interpolation between bracketing samples,
  `playback.py:62`), while **gear and DRS use `_sample_hold`**
  (last-value hold, `playback.py:76`) — interpolating `G3→G4` through 3.5
  or DRS `0→8` through 4 would display meaningless intermediate values.
- **Ranking** (`playback.py:163-170`): sort by `-race_dist`; but once
  `t ≥ finished_at`, sort by the official classification instead — drivers
  cross the line at different times and the live order must freeze into the
  real result.
- **Lap display**: `lap_current` = max lap over active (non-OUT) drivers
  with `lap > 0` (`playback.py:174-175`).
- **SC anchor**: `compute_sc` needs the leader's *track* position `s`
  (not race distance) — `_leader_track_s` (`playback.py:216`) samples it.

`FrameState` (`playback.py:44`) is the immutable per-frame contract between
the engine and every renderer: 20 `DriverFrame`s with rank/position, lap,
compound, DRS text, OUT flag, plus status text and the optional `SCFrame`.

---

## 5. Safety Car simulation (`safety_car.py`)

FastF1 publishes **no GPS for the Safety Car itself** — only status codes.
The SC is therefore *simulated* from real data (`compute_sc`,
`safety_car.py:32`), keyed on each `SCWindow`:

```
target = (leader_s + 500 m) mod L          # SC runs 500 m up the road

phase 1  deploy   [start, start+3s)        # burst from the pit-lane reference
         s = pit_exit_s + f · shortest_delta(pit_exit_s → target)
         xy blended from pit_lane_pts[-1] → track point as f → 1
         label "SC DEPLOYING", soft pulsing glow

phase 2  track    [start+3s, end]          # circulate ahead of the leader
         xy = point_at(target)  (re-evaluated every frame → follows leader)
         label "SC", full glow

phase 3  return   (end, end+3s]            # hand back and box
         s = target' + f · shortest_delta(target' → pit_entry_s)
         xy blends toward pit_lane_pts[0]
         label "SC IN", fading glow
```

- `DEPLOY_DURATION = RETURN_DURATION = 3` **replay seconds**
  (`safety_car.py:13-14`) — compressed theatre, not real-world timing.
- `_shortest_delta` (`safety_car.py:27`) is the signed shortest path around
  the loop: `(b − a + 1.5·L) mod L − 0.5·L` — so the SC never drives the
  "wrong way round" to reach its target.
- Rendering adds phase-dependent alpha and a pulsing ring
  (`track_render.py:180`).

---

## 6. Rendering

### 6.1 `theme.py` — shared primitives

- `font(size, bold)` is `lru_cache`d (`theme.py:48`) — `SysFont` lookups
  happen once per size.
- `glow_template` (`theme.py:116`) pre-renders 3 concentric alpha rings per
  `(radius, color)`; `blit_glow` (`theme.py:128`) applies per-frame alpha
  and optional smoothscaled pulse. Glow for selection and SC costs one
  blit each, not per-ring draws.
- `panel()` draws a rounded translucent surface + 1 px edge; `format_clock`
  / `format_laptime` produce `1:23:45` and `1:33.370`.

### 6.2 `track_render.py` — the track view

**Fit transform** (`layout`, `track_render.py:28`): track bounding box +
34 px margin scaled to the viewport with uniform `scale`, **y flipped**
(world north → screen up), centred. `world_to_screen` is two multiplies.

**Static layer** (`_rebuild_static`, `track_render.py:58`) — everything that
doesn't change per frame is pre-rendered onto one surface **once per layout
or toggle**:

1. background fill;
2. track: wide `TRACK_EDGE` stroke under a narrower `TRACK` stroke
   (width `max(10, 13·scale)` px — track *width* in metres isn't drawn to
   scale, it's a readability choice);
3. DRS zones: slice vertex arrays between `cum`-searchsorted bounds for
   each zone (`_zone_points`, line 103, seam-aware) → thick green arcs;
4. pit-lane polyline, thin, dim;
5. start/finish: 6 m chord across the track, perpendicular to the tangent;
6. corner numbers as dim 11 px labels.

Each frame then: blit static layer → draw 20 cars **in reverse rank order**
(P1 last = on top) → name labels → SC → clip to viewport
(`draw`, `track_render.py:120`).

**Cars**: 6 px team-colour disc + dark outline; selection adds a pulsing
white glow + 2 px ring; OUT cars are desaturated and crossed with a red ✕.
Labels render twice (dark underlay at +1,−1, then the text) for legibility
without a font stroke.

**Hit-testing**: `draw` records each car's *screen* position into
`car_screen`; `pick()` (`track_render.py:203`) returns the nearest number
within 13 px — O(D) per click, trivial at 20 drivers.

### 6.3 `hud.py` — chrome

Layout bands (`layout`, `hud.py:28`): top bar 56 px, leaderboard 300 px
right, transport strip 96 px bottom (56 px when the progress bar is
hidden), viewport takes the rest.

- **Top bar** (`hud.py:57`): session title, `LAP n/m`, race clock
  `elapsed / total`, and a status pill (colour from
  `theme.STATUS_COLORS`) with an extra ring while `in_sc_window`.
- **Leaderboard** (`hud.py:81`): rows from `FrameState.drivers` already in
  rank order; rank, team-colour chip, code, surname, tyre badge
  (compound colour + letter), red `OUT`; selected row gets highlight +
  accent bar, hovered row a subtle lift. Rects are returned for
  click-through selection.
- **Transport** (`hud.py:137`): rewind / play / forward / speed / restart
  buttons + progress track. SC windows paint as amber spans; the handle is
  a white dot at `frac = (t − t0) / duration`; clicking anywhere in the
  inflated region scrubs (`App._scrub_to`).
- **Legend** (`hud.py:212`): static controls reference anchored above the
  transport strip.

### 6.4 `panels.py` — floating insights

`FloatingPanel` (`panels.py:13`) is a small state machine:

- `begin_drag(pos)` returns `'close' | 'drag' | None`
  (`panels.py:31`) — close clicks **never** start a drag (a past bug, §10);
- `_drag_off` is initialised to `None` and guarded in `drag_to`
  (`panels.py:41`); header presses store the grab offset;
- `App` owns the drag lifecycle: press → `dragging = panel`, motion →
  `drag_to`, release → `end_drag`.

Panels: **InsightsMenu** (checkbox list, `panels.py:63`), **TelemetryPanel**
(speed/gear/DRS/compound per selected driver, auto-height,
`panels.py:92`), **LapChartPanel** (`panels.py:144` — per-driver lap-time
lines; NaN and `> 1.6 × min` filtered to kill SC-lap outliers; axis ticks
every `max(5, laps//5)`; re-sorted only when the selection set changes),
**StintsPanel** (`panels.py:229` — compound bars across lap axis).
All are dragged by their header, closed by their ✕, and rendered on demand
each frame (chart cost is ~50 points per selected driver).

---

## 7. Orchestration (`app.py`)

- **Event flow** (`handle_events`, `app.py:53`): events are pulled once
  per frame and routed through `_dispatch` (`app.py:65`) inside a
  `try/except` that prints the traceback and *skips* the event — a bad
  input event can never kill the window (§10).
- **Shift state** for multi-select comes from `pygame.key.get_mods()`
  (`app.py:79`), **not** the event — real SDL mouse events carry no
  `mod` attribute.
- **Click routing** (`on_click`, `app.py:135`), in order: visible panels
  (topmost first: close/drag) → insights menu items → transport buttons →
  progress region (start scrub) → leaderboard rows → viewport
  (car pick, or clear selection on empty space).
- **Selection semantics** (`pick_number`): click selects exclusively, plain
  re-click of the sole selection clears, shift toggles membership.
- **State machines**: `dragging: FloatingPanel | None` and
  `scrubbing: bool`, both reset on button-up; held ←/→ keys scrub at
  45× replay speed inside `update()` (`app.py:215`).
- **Menu truth**: `sync_states()` (`app.py:130`) mirrors panel `.visible`
  into the checkbox dict before the menu draws, so closing a panel with its
  ✕ keeps the menu honest.
- **Draw order** (`draw`, `app.py:228`): layout → top bar → track view →
  leaderboard → transport → legend → chart → stints → telemetry → menu
  (menu last = topmost).
- **Smoke mode** (`smoke`, `app.py:269`): seeks to five representative
  timestamps (start / running / SC / mid / finish), renders each to PNG —
  the whole app with a window, under `SDL_VIDEODRIVER=dummy`.

---

## 8. CLI (`main.py`)

- `parse_args` (`main.py:34`): `--year/--gp/--session` (defaults:
  2023 / Las Vegas / `R`), `--fullscreen`, `--rebuild`,
  `--smoke-test [DIR]`, `--list-gps [YEAR]`, `--list-cache`,
  `--clear-cache`.
- `resolve_session` (`main.py:64`) maps aliases: `race→R`,
  `qualifying→Q`, `sprint→S`, `shootout→SQ`, `fp1→FP1`, …
- `list_gps` (`main.py:81`) prints `fastf1.get_event_schedule(year)` —
  round, canonical `EventName` (copy-paste ready for `--gp`), date, and
  abbreviated sessions (`FP1…R`, `SQ S` on sprint weekends). Canonical
  names matter: FastF1's fuzzy matching once resolved `"great britain"` to
  the *Austrian* GP.
- `main()` (`main.py:140`) orders operations so non-GUI commands exit
  before any window opens; the GUI path shows a loading window driven by a
  `progress` callback while `load_dataset` runs.

---

## 9. Performance

| Phase | Cost | Once/… |
| --- | --- | --- |
| FastF1 download | 30–60 s network | session lifetime |
| `build_dataset` | ~8 s (KD projections dominate) | session lifetime |
| pickle load | ~1 s (30–50 MB) | app start |
| `frame()` | 20 × `searchsorted` + sort + SC math ≪ 1 ms | frame |
| static layer blit | 1 big blit + 20 discs + labels ≈ 1 ms | frame |
| panel redraws | ~10 font renders (cached fonts) | frame |

Per-frame work is **O(D log N)** sampling plus constant-size drawing;
nothing scales with session length because all arrays are precomputed and
time-sliced by binary search. Memory: arrays are float32, a full race
pickle is 30–52 MB on disk.

Caches at every layer: FastF1 HTTP sqlite, preprocessed pickle, `lru_cache`
fonts, `lru_cache` glow templates, per-layout static track surface, and
chart series keyed on the selection tuple.

---

## 10. Postmortem: three real bugs

Each escaped into the running app and was found by observation or
real-display testing — not by unit tests, which is the lesson.

### 10.1 Phantom laps (data correctness)

**Symptom**: drivers gained laps that never happened; the leaderboard
diverged from reality after pit stops.
**Cause**: lap-counting by `s`-wraps; pit-lane and S/F seam crossings
counted as extra lap completions.
**Fix**: integrate signed deltas with ±L/2 folding (`_integrate_progress`,
`preprocess.py:140`), ranking on continuous `race_dist`.
**Guard**: cross-checked `race_dist/length ≈ official lap count` for all
finishers of two races before/after the fix.

### 10.2 One giant DRS zone (heuristic calibration)

**Symptom**: after switching to a relative threshold during Vegas
development, a single ~5 km "DRS zone" spanned most of the track.
**Cause**: absolute minimum-bin count interacted badly with busy circuits —
noise bins cleared the floor everywhere.
**Fix**: require ≥4% of the peak bin *and* ≥3 samples, drop runs <80 m,
merge gaps ≤200 m (`preprocess.py:435-470`). Verified: Vegas → two zones
on the long straights.

### 10.3 Two crashes that killed the window (input handling)

1. **`TypeError: 'NoneType' object is not subscriptable`**
   — clicking a panel's ✕ set `dragging = panel` without an offset; the
   next mouse motion did `None[0]`. Fixed by making `begin_drag` return
   `'close' | 'drag' | None` (`panels.py:31`) — close never drags — and
   guarding `drag_to` (`panels.py:41`).
2. **`AttributeError: 'pygame.event.Event' object has no attribute 'mod'`**
   — reading `ev.mod` from mouse events; SDL only attaches `mod` to
   keyboard events. **Why tests missed it**: the headless suite posted
   synthetic mouse events *with* `mod`. Fixed by reading
   `pygame.key.get_mods()` (`app.py:79`); the test now posts
   real-shaped events (`tests/test_controls.py:42`), and a per-event
   safety net (`app.py:53`) means any future input bug logs a traceback
   instead of closing the app.

Both have explicit regression steps in `tests/test_controls.py`.

---

## 11. Known limitations & possible extensions

- **SC timing is compressed** (3 s deploy/return) — theatrical, not
  historical. Real pit-out/queue timing would need race-control messages.
- **DRS detection is heuristic** — works on data-rich modern seasons; DRS
  availability codes vary across eras.
- **No weather/flag overlays**, no fastest-lap flash, no interval gaps.
  `FrameState` already carries everything needed for a gap tower.
- **One track reference per session** — no multi-session track sharing
  yet (qualifying + race rebuild independently; keys differ, so cache
  doubles).
- **Session types**: tested on races; qualifying/sprint should work but
  `finished_at`/`total_laps` semantics differ (no chequered-flag freeze in
  Q).
- Natural next steps: gap/interval tower, race-control message ticker,
  per-corner speed traces, in-app session switcher.

---

*See also: [README](../README.md) for installation, usage and controls.*
