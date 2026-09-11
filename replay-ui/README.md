# GeoTraceLab Pacman replay

Read-only visualization of the existing production/non-oracle Pacman output.
The production panel currently opens trip 07-22; the 07-26 replay is retained
as a second exported dataset.
The exporter never runs or imports the tracker and never feeds withheld GPS back into it.

## Run

```bash
cd replay-ui
npm install
npm run data
npm run dev
```

Open <http://localhost:3000/>.

The generated replay files are committed at `public/data/replay-07-22.json` and
`public/data/replay-07-26.json`, so regeneration is only necessary after source
artifacts change. A production build can be checked with `npm run build`.

## Data flow

- Pacman coordinates: `runs/pacman-display/2026-07-22/track.geojson`, feature `pacman_top1`.
- Pacman edge, distance-along-edge, hypothesis id, estimated speed, and estimated distance:
  `runs/pacman-display/2026-07-22/frames.json`.
- Ground-truth coordinates and speed: withheld `runs/review-final/2026-07-22/trip/reference-samples.jsonl`.
- Ground-truth edge: `runs/pacman-display/2026-07-22/ground_truth_trace.json`.
- Display basemap: keyless OpenStreetMap raster tiles from `openstreetmap.de`, rendered with Leaflet.
- Tracker road network (diagnostic metadata only): `runs/review-map.graphml`.

`scripts/export_replay.py` converts those artifacts into normalized visualization frames.
`app/page.tsx` consumes only that normalized structure, which can later be supplied by a live
adapter instead of this replay file.

Pacman is rendered from the selected production-safe `single_path` output. The
renderer does not select hypotheses or reconstruct a route of its own. The UI
reports current-edge agreement separately from committed-route topology.
