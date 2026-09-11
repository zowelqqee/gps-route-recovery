'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import type {
  CircleMarker,
  Map as LeafletMap,
  Marker,
  Polyline,
  TileLayer,
} from 'leaflet';
import {
  Crosshair,
  Eye,
  EyeOff,
  Gauge,
  Map as MapIcon,
  Pause,
  Play,
  RotateCcw,
  Route,
  Satellite,
} from 'lucide-react';

import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';

type Point = {
  lon: number;
  lat: number;
  edgeId: number;
  s?: number;
  hypothesisId?: number;
};
type ReplayFrame = {
  t: number;
  elapsed: number;
  timestamp: string;
  tracker: Point;
  route?: Point;
  truth: Point;
  vEst: number;
  vRoute?: number;
  vTrue: number;
  dEst: number;
  dRoute?: number;
  dTrue: number;
  alongError: number;
  routeAlongError?: number;
  gpsAvailable: boolean;
  edgeMatch: boolean;
  topologyMatch: boolean;
  display?: {
    estimator: string;
    gateActive: boolean;
    deltaM: number;
    excessM: number;
    atFrontier: boolean;
    displayError: number;
  };
};
type ReplayData = {
  schemaVersion: number;
  meta: {
    tripId: string;
    tripLabel: string;
    mode: string;
    trackerMode: string;
    outageStartT: number;
    outageDuration: number;
    source: Record<string, string>;
    report: {
      meanPositionErrorM: number;
      maxPositionErrorM: number;
      finalAlongErrorM: number;
      decisionCount?: number;
      realWrongDecisionCount?: number | null;
      twinEdgeArtifactCount?: number;
      checkpointCount?: number;
      checkpointFinalErrorM?: number;
    };
    referenceKind?: string;
    rfid?: {
      checkpointCount: number;
      checkpoints: Array<{
        anchorId: string;
        elapsed: number;
        markerErrorM: number;
        estimatedDistanceM: number;
        referenceDistanceM: number;
      }>;
      bindingResidualMedianMsec: number | null;
      physicalTimestampUsed: boolean;
      note: string;
    };
    display?: {
      estimator: string;
      note: string;
      leave0726Out: boolean;
      gateActiveFraction: number;
      maxAbsDeltaM: number;
      displayError: { medianM: number; p95M: number; maxM: number };
      routeError: { medianM: number; p95M: number; maxM: number };
      closerThanRouteFraction: number;
    } | null;
  };
  bounds: [[number, number], [number, number]];
  frames: ReplayFrame[];
};
type Layers = {
  pacman: boolean;
  trail: boolean;
  truth: boolean;
  basemap: boolean;
  diagnostics: boolean;
};

type WebMcpContext = {
  registerTool: (
    tool: {
      name: string;
      title: string;
      description: string;
      inputSchema: Record<string, unknown>;
      annotations: { readOnlyHint: boolean; untrustedContentHint: boolean };
      execute: (input: unknown) => unknown;
    },
    options?: { signal?: AbortSignal },
  ) => void | Promise<void>;
};

const SPEEDS = [0.5, 1, 2, 5];
const TRIPS = [
  '07-22',
  '07-23',
  '07-24',
  '07-25',
  '07-26',
  '09-08-rfid',
] as const;
type TripId = (typeof TRIPS)[number];
function formatElapsed(value: number) {
  const whole = Math.max(0, Math.floor(value));
  return `${String(Math.floor(whole / 60)).padStart(2, '0')}:${String(whole % 60).padStart(2, '0')}`;
}

function formatTimestamp(value: string) {
  return `${new Date(value).toISOString().slice(11, 19)} UTC`;
}

function formatDistance(value: number) {
  return Math.abs(value) >= 1000
    ? `${(value / 1000).toFixed(2)} km`
    : `${value.toFixed(1)} m`;
}

function markerDistanceM(a: Point, b: Point) {
  const earthRadiusM = 6_371_008.8;
  const toRadians = Math.PI / 180;
  const lat1 = a.lat * toRadians;
  const lat2 = b.lat * toRadians;
  const deltaLat = (b.lat - a.lat) * toRadians;
  const deltaLon = (b.lon - a.lon) * toRadians;
  const haversine =
    Math.sin(deltaLat / 2) ** 2 +
    Math.cos(lat1) * Math.cos(lat2) * Math.sin(deltaLon / 2) ** 2;
  return 2 * earthRadiusM * Math.asin(Math.min(1, Math.sqrt(haversine)));
}

function findFrameIndex(frames: ReplayFrame[], elapsed: number) {
  let lo = 0;
  let hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].elapsed <= elapsed) lo = mid;
    else hi = mid - 1;
  }
  return lo;
}

function interpolatePoint(a: Point, b: Point, mix: number): Point {
  return {
    lon: a.lon + (b.lon - a.lon) * mix,
    lat: a.lat + (b.lat - a.lat) * mix,
    edgeId: a.edgeId,
    s: a.s,
  };
}

function ReplayMap({
  data,
  playhead,
  layers,
}: {
  data: ReplayData;
  playhead: number;
  layers: Layers;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<LeafletMap | null>(null);
  const leafletRef = useRef<typeof import('leaflet') | null>(null);
  const tileLayerRef = useRef<TileLayer | null>(null);
  const trailRef = useRef<Polyline | null>(null);
  const truthTrailRef = useRef<Polyline | null>(null);
  const trackerMarkerRef = useRef<Marker | null>(null);
  const truthMarkerRef = useRef<CircleMarker | null>(null);
  const [mapReady, setMapReady] = useState(false);
  const [autoFollow, setAutoFollow] = useState(true);
  const index = findFrameIndex(data.frames, playhead);
  const nextIndex = Math.min(data.frames.length - 1, index + 1);
  const current = data.frames[index];
  const next = data.frames[nextIndex];
  const span = Math.max(0.001, next.elapsed - current.elapsed);
  const mix = Math.min(1, Math.max(0, (playhead - current.elapsed) / span));
  const tracker = interpolatePoint(current.tracker, next.tracker, mix);
  const truth = interpolatePoint(current.truth, next.truth, mix);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    let cancelled = false;
    const stopFollowing = () => setAutoFollow(false);
    void import('leaflet').then((L) => {
      if (cancelled || mapRef.current) return;
      const initialTracker = data.frames[0].tracker;
      leafletRef.current = L;
      const map = L.map(container, {
        zoomControl: true,
        attributionControl: true,
        preferCanvas: true,
        zoomAnimation: false,
        fadeAnimation: false,
        markerZoomAnimation: false,
      });
      const tiles = L.tileLayer(
        'https://tile.openstreetmap.de/{z}/{x}/{y}.png',
        {
          maxZoom: 19,
          attribution:
            '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
        },
      ).addTo(map);
      const truthTrail = L.polyline([], {
        color: '#36d3c7',
        weight: 3,
        opacity: 0.72,
        dashArray: '7 7',
        lineJoin: 'round',
      }).addTo(map);
      const trail = L.polyline([], {
        color: '#f4c84d',
        weight: 4,
        opacity: 0.9,
        lineJoin: 'round',
      }).addTo(map);
      const trackerMarker = L.marker([initialTracker.lat, initialTracker.lon], {
        interactive: false,
        zIndexOffset: 1000,
        icon: L.divIcon({
          className: 'pacman-leaflet-icon',
          html: '<span class="pacman-marker" />',
          iconSize: [30, 30],
          iconAnchor: [15, 15],
        }),
      }).addTo(map);
      const truthMarker = L.circleMarker(
        [data.frames[0].truth.lat, data.frames[0].truth.lon],
        {
          radius: 7,
          color: '#d8fffb',
          weight: 2,
          fillColor: '#36d3c7',
          fillOpacity: 1,
        },
      ).addTo(map);
      mapRef.current = map;
      tileLayerRef.current = tiles;
      trailRef.current = trail;
      truthTrailRef.current = truthTrail;
      trackerMarkerRef.current = trackerMarker;
      truthMarkerRef.current = truthMarker;
      map.setView([initialTracker.lat, initialTracker.lon], 16);
      map.on('dragstart', stopFollowing);
      container.addEventListener('wheel', stopFollowing, { passive: true });
      setMapReady(true);
      requestAnimationFrame(() => map.invalidateSize());
    });
    return () => {
      cancelled = true;
      container.removeEventListener('wheel', stopFollowing);
      setMapReady(false);
      const map = mapRef.current;
      if (map) {
        map.stop();
        map.off();
        map.remove();
      }
      mapRef.current = null;
      leafletRef.current = null;
      tileLayerRef.current = null;
      trailRef.current = null;
      truthTrailRef.current = null;
      trackerMarkerRef.current = null;
      truthMarkerRef.current = null;
    };
  }, [data]);

  useEffect(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    const trail = trailRef.current;
    const truthTrail = truthTrailRef.current;
    const trackerMarker = trackerMarkerRef.current;
    const truthMarker = truthMarkerRef.current;
    if (
      !mapReady ||
      !map ||
      !L ||
      !trail ||
      !truthTrail ||
      !trackerMarker ||
      !truthMarker
    )
      return;

    const trackerLatLng: [number, number] = [tracker.lat, tracker.lon];
    const truthLatLng: [number, number] = [truth.lat, truth.lon];
    trail.setLatLngs([
      ...data.frames
        .slice(0, index + 1)
        .map(
          (frame) => [frame.tracker.lat, frame.tracker.lon] as [number, number],
        ),
      trackerLatLng,
    ]);
    truthTrail.setLatLngs([
      ...data.frames
        .slice(0, index + 1)
        .map((frame) => [frame.truth.lat, frame.truth.lon] as [number, number]),
      truthLatLng,
    ]);
    trackerMarker.setLatLng(trackerLatLng);
    truthMarker.setLatLng(truthLatLng);

    const dLon =
      (next.tracker.lon - tracker.lon) *
      Math.cos((tracker.lat * Math.PI) / 180);
    const dLat = next.tracker.lat - tracker.lat;
    const angle = Math.atan2(-dLat, dLon) * (180 / Math.PI);
    trackerMarker.setIcon(
      L.divIcon({
        className: 'pacman-leaflet-icon',
        html: `<span class="pacman-marker" style="transform:rotate(${angle.toFixed(1)}deg)" />`,
        iconSize: [30, 30],
        iconAnchor: [15, 15],
      }),
    );

    if (autoFollow) {
      const paddedView = map.getBounds().pad(-0.2);
      if (
        !paddedView.contains(trackerLatLng) ||
        !paddedView.contains(truthLatLng)
      ) {
        map.fitBounds(L.latLngBounds([trackerLatLng, truthLatLng]), {
          padding: [90, 90],
          maxZoom: 16,
          animate: false,
        });
      }
    }
  }, [
    autoFollow,
    data.frames,
    index,
    mapReady,
    next.tracker.lat,
    next.tracker.lon,
    tracker.lat,
    tracker.lon,
    truth.lat,
    truth.lon,
  ]);

  useEffect(() => {
    const map = mapRef.current;
    const tiles = tileLayerRef.current;
    if (!mapReady || !map || !tiles) return;
    if (layers.basemap && !map.hasLayer(tiles)) tiles.addTo(map);
    if (!layers.basemap && map.hasLayer(tiles)) tiles.removeFrom(map);
  }, [layers.basemap, mapReady]);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map) return;
    const trail = trailRef.current;
    const truthTrail = truthTrailRef.current;
    const trackerMarker = trackerMarkerRef.current;
    const truthMarker = truthMarkerRef.current;
    if (trail) {
      if (layers.trail) trail.addTo(map);
      else trail.removeFrom(map);
    }
    if (trackerMarker) {
      if (layers.pacman) trackerMarker.addTo(map);
      else trackerMarker.removeFrom(map);
    }
    if (truthMarker) {
      if (layers.truth) {
        truthTrail?.addTo(map);
        truthMarker.addTo(map);
      } else {
        truthTrail?.removeFrom(map);
        truthMarker.removeFrom(map);
      }
    }
  }, [layers.pacman, layers.trail, layers.truth, mapReady]);

  const recenter = useCallback(() => {
    const map = mapRef.current;
    const L = leafletRef.current;
    if (!map || !L) return;
    setAutoFollow(true);
    map.fitBounds(
      L.latLngBounds([
        [tracker.lat, tracker.lon],
        [truth.lat, truth.lon],
      ]),
      { padding: [90, 90], maxZoom: 16, animate: false },
    );
  }, [tracker.lat, tracker.lon, truth.lat, truth.lon]);

  return (
    <div className="map-shell">
      <div
        ref={containerRef}
        className="map-canvas"
        aria-label="OpenStreetMap comparing Pacman tracker and ground truth positions"
      />
      <div className="map-statuses">
        <div
          className={`status-chip ${current.gpsAvailable ? 'gps-on' : 'gps-off'}`}
        >
          <Satellite size={14} />{' '}
          {current.gpsAvailable ? 'GPS AVAILABLE' : 'GPS OUTAGE'}
        </div>
        <div
          className={`status-chip ${current.topologyMatch ? 'topology-on' : 'topology-off'}`}
        >
          <Route size={14} />{' '}
          {current.topologyMatch
            ? current.edgeMatch
              ? 'SAME ROAD NOW'
              : 'SAME ROUTE · POSITION OFFSET'
            : 'ROUTE DIVERGED'}
        </div>
      </div>
      <Button
        className={`follow-button ${autoFollow ? 'is-active' : ''}`}
        variant="outline"
        size="sm"
        onClick={recenter}
      >
        <Crosshair size={14} /> {autoFollow ? 'Following' : 'Re-center'}
      </Button>
      <div className="map-legend">
        <span>
          <i className="legend-pacman" /> Pacman
        </span>
        <span>
          <i className="legend-truth" />{' '}
          {data.meta.rfid ? 'RFID route reference' : 'Ground truth'}
        </span>
      </div>
    </div>
  );
}

function ErrorChart({
  frames,
  index,
  onSeek,
}: {
  frames: ReplayFrame[];
  index: number;
  onSeek: (value: number) => void;
}) {
  const width = 300;
  const height = 96;
  const distances = frames.map((frame) =>
    markerDistanceM(frame.tracker, frame.truth),
  );
  const maximum = Math.max(100, ...distances);
  const points = frames
    .map(
      (_frame, i) =>
        `${((i / (frames.length - 1)) * width).toFixed(1)},${(height - 8 - (distances[i] / maximum) * (height - 16)).toFixed(1)}`,
    )
    .join(' ');
  const cursorX = (index / (frames.length - 1)) * width;
  const cursorY = height - 8 - (distances[index] / maximum) * (height - 16);
  return (
    <button
      type="button"
      className="error-chart"
      aria-label="Seek by Pacman to ground-truth marker distance chart"
      onClick={(event) => {
        const rect = event.currentTarget.getBoundingClientRect();
        const ratio = Math.min(
          1,
          Math.max(0, (event.clientX - rect.left) / rect.width),
        );
        onSeek(ratio * frames[frames.length - 1].elapsed);
      }}
    >
      <svg
        viewBox={`0 0 ${width} ${height}`}
        preserveAspectRatio="none"
        aria-hidden="true"
      >
        <line
          x1="0"
          y1={height - 8}
          x2={width}
          y2={height - 8}
          className="chart-zero"
        />
        <polyline points={points} className="chart-line" />
        <line
          x1={cursorX}
          y1="4"
          x2={cursorX}
          y2={height - 4}
          className="chart-cursor"
        />
        <circle cx={cursorX} cy={cursorY} r="3.5" className="chart-dot" />
      </svg>
      <span className="chart-limit plus">{Math.round(maximum)} m</span>
      <span className="chart-limit minus">0 m</span>
    </button>
  );
}

function LayerToggle({
  checked,
  label,
  accent,
  onChange,
}: {
  checked: boolean;
  label: string;
  accent?: string;
  onChange: (value: boolean) => void;
}) {
  return (
    <label className="layer-toggle">
      <span>
        {accent && <i style={{ background: accent }} />} {label}
      </span>
      <Switch
        checked={checked}
        onCheckedChange={onChange}
        aria-label={`Toggle ${label}`}
      />
    </label>
  );
}

export default function Home() {
  const [selectedTrip, setSelectedTrip] = useState<TripId>('07-22');
  const [data, setData] = useState<ReplayData | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [playhead, setPlayhead] = useState(0);
  const [layers, setLayers] = useState<Layers>({
    pacman: true,
    trail: true,
    truth: true,
    basemap: true,
    diagnostics: true,
  });
  const lastTick = useRef<number | null>(null);
  const playheadRef = useRef(playhead);
  const speedRef = useRef(speed);
  const playingRef = useRef(playing);

  useEffect(() => {
    playheadRef.current = playhead;
    speedRef.current = speed;
    playingRef.current = playing;
  }, [playhead, speed, playing]);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/data/replay-${selectedTrip}.json`, { signal: controller.signal })
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json() as Promise<ReplayData>;
      })
      .then(setData)
      .catch((error: Error) => {
        if (error.name !== 'AbortError') setLoadError(error.message);
      });
    return () => controller.abort();
  }, [selectedTrip]);

  useEffect(() => {
    if (!playing || !data) {
      lastTick.current = null;
      return;
    }
    let request = 0;
    const tick = (now: number) => {
      const before = lastTick.current ?? now;
      lastTick.current = now;
      setPlayhead((value) => {
        const next = value + ((now - before) / 1000) * speed;
        if (next >= data.meta.outageDuration) {
          setPlaying(false);
          return data.meta.outageDuration;
        }
        return next;
      });
      request = requestAnimationFrame(tick);
    };
    request = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(request);
  }, [data, playing, speed]);

  useEffect(() => {
    if (!data) return;
    const context = (document as Document & { modelContext?: WebMcpContext })
      .modelContext;
    if (!context?.registerTool) return;
    const lifecycle = new AbortController();
    const register = (tool: Parameters<WebMcpContext['registerTool']>[0]) => {
      try {
        void Promise.resolve(
          context.registerTool(tool, { signal: lifecycle.signal }),
        ).catch(() => undefined);
      } catch {
        // WebMCP is optional in browsers that expose a partial implementation.
      }
    };

    register({
      name: 'seek_replay',
      title: 'Seek replay',
      description:
        'Move the visible Pacman, ground-truth marker, trail, and diagnostics to an outage elapsed time.',
      inputSchema: {
        type: 'object',
        properties: {
          elapsedSeconds: {
            type: 'number',
            minimum: 0,
            maximum: data.meta.outageDuration,
          },
        },
        required: ['elapsedSeconds'],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute(input) {
        const value = (input as { elapsedSeconds?: unknown })?.elapsedSeconds;
        if (
          typeof value !== 'number' ||
          !Number.isFinite(value) ||
          value < 0 ||
          value > data.meta.outageDuration
        ) {
          throw new Error(
            `elapsedSeconds must be between 0 and ${data.meta.outageDuration}`,
          );
        }
        setPlayhead(value);
        playheadRef.current = value;
        return {
          elapsedSeconds: value,
          frameIndex: findFrameIndex(data.frames, value),
        };
      },
    });

    register({
      name: 'set_replay_transport',
      title: 'Set replay transport',
      description:
        'Play, pause, or restart the visible replay and optionally set its playback speed.',
      inputSchema: {
        type: 'object',
        properties: {
          action: { type: 'string', enum: ['play', 'pause', 'restart'] },
          speed: { type: 'number', enum: SPEEDS },
        },
        required: ['action'],
        additionalProperties: false,
      },
      annotations: { readOnlyHint: false, untrustedContentHint: false },
      execute(input) {
        const value = input as { action?: unknown; speed?: unknown };
        if (!['play', 'pause', 'restart'].includes(String(value?.action)))
          throw new Error('action must be play, pause, or restart');
        if (value.speed !== undefined && !SPEEDS.includes(Number(value.speed)))
          throw new Error('speed must be 0.5, 1, 2, or 5');
        if (value.speed !== undefined) {
          setSpeed(Number(value.speed));
          speedRef.current = Number(value.speed);
        }
        if (value.action === 'restart') {
          setPlayhead(0);
          playheadRef.current = 0;
          setPlaying(false);
          playingRef.current = false;
        } else {
          const shouldPlay = value.action === 'play';
          setPlaying(shouldPlay);
          playingRef.current = shouldPlay;
        }
        return {
          action: value.action,
          elapsedSeconds: playheadRef.current,
          speed: speedRef.current,
          playing: playingRef.current,
        };
      },
    });

    return () => lifecycle.abort();
  }, [data]);

  if (loadError)
    return (
      <main className="state-page">
        Replay data could not be loaded: {loadError}
      </main>
    );
  if (!data)
    return (
      <main className="state-page">
        <span className="loading-dot" /> Loading trip {selectedTrip}…
      </main>
    );

  const index = findFrameIndex(data.frames, playhead);
  const frame = data.frames[index];
  const markerGapM = markerDistanceM(frame.tracker, frame.truth);
  const duration = data.meta.outageDuration;
  const selectTrip = (trip: TripId) => {
    setData(null);
    setLoadError(null);
    setPlaying(false);
    setPlayhead(0);
    playheadRef.current = 0;
    playingRef.current = false;
    lastTick.current = null;
    setSelectedTrip(trip);
  };
  const toggleLayer = (key: keyof Layers) => (value: boolean) =>
    setLayers((current) => ({ ...current, [key]: value }));

  return (
    <main className="app-shell">
      <header className="topbar">
        <div className="brand-lockup">
          <div className="brand-mark">
            <MapIcon size={17} />
          </div>
          <div>
            <div className="eyebrow">GEOTRACELAB</div>
            <h1>Pacman Tracker</h1>
          </div>
        </div>
        <div className="run-context">
          <label className="trip-picker">
            <span>TRIP</span>
            <select
              value={selectedTrip}
              onChange={(event) => selectTrip(event.target.value as TripId)}
              aria-label="Select replay trip"
            >
              {TRIPS.map((trip) => (
                <option key={trip} value={trip}>
                  {trip === '09-08-rfid' ? '09-08 · RFID' : trip}
                </option>
              ))}
            </select>
          </label>
          <span>Full outage · {formatElapsed(duration)}</span>
          <span className="context-divider" />
          <span className="source-pill">
            {data.meta.trackerMode.toUpperCase()}
          </span>
          {data.meta.rfid && (
            <>
              <span className="context-divider" />
              <span className="source-pill" title={data.meta.rfid.note}>
                SPARSE RFID · {data.meta.rfid.checkpointCount} CHECKPOINTS
              </span>
            </>
          )}
          {data.meta.display && (
            <>
              <span className="context-divider" />
              <span
                className="source-pill"
                title={data.meta.display.note}
              >
                DISPLAY · {data.meta.display.estimator} · median{' '}
                {formatDistance(data.meta.display.displayError.medianM)} vs route{' '}
                {formatDistance(data.meta.display.routeError.medianM)}
              </span>
            </>
          )}
        </div>
        <div className="live-indicator">
          <span /> REPLAY
        </div>
      </header>

      <section
        className={`workspace ${layers.diagnostics ? '' : 'diagnostics-hidden'}`}
      >
        <div className="map-column">
          <ReplayMap data={data} playhead={playhead} layers={layers} />
          <section className="transport" aria-label="Replay controls">
            <div className="transport-row">
              <Button
                className="play-button"
                size="icon-lg"
                onClick={() => {
                  if (playhead >= duration) setPlayhead(0);
                  setPlaying((value) => !value);
                }}
                aria-label={playing ? 'Pause replay' : 'Play replay'}
              >
                {playing ? (
                  <Pause size={18} fill="currentColor" />
                ) : (
                  <Play size={18} fill="currentColor" />
                )}
              </Button>
              <Button
                className="restart-button"
                size="icon-lg"
                variant="outline"
                onClick={() => {
                  setPlaying(false);
                  setPlayhead(0);
                }}
                aria-label="Restart replay"
              >
                <RotateCcw size={17} />
              </Button>
              <div className="time-readout">
                <strong>{formatElapsed(playhead)}</strong>
                <span>/ {formatElapsed(duration)}</span>
              </div>
              <div className="timeline-wrap">
                <Slider
                  min={0}
                  max={duration}
                  step={0.05}
                  value={[playhead]}
                  onValueChange={(value) =>
                    setPlayhead(typeof value === 'number' ? value : value[0])
                  }
                  aria-label="Replay timeline"
                />
                <div className="timeline-labels">
                  <span>OUTAGE START</span>
                  <span>{formatTimestamp(frame.timestamp)}</span>
                  <span>END</span>
                </div>
              </div>
              <div className="speed-control" aria-label="Playback speed">
                {SPEEDS.map((rate) => (
                  <button
                    type="button"
                    key={rate}
                    className={speed === rate ? 'active' : ''}
                    onClick={() => setSpeed(rate)}
                  >
                    {rate}×
                  </button>
                ))}
              </div>
            </div>
          </section>
        </div>

        {layers.diagnostics && (
          <aside className="diagnostics-panel">
            <section className="diag-section clock-section">
              <div className="section-kicker">OUTAGE ELAPSED</div>
              <div className="big-clock">
                {formatElapsed(playhead)}
                <small>.{Math.floor((playhead % 1) * 10)}</small>
              </div>
              <div className="timestamp">
                {formatTimestamp(frame.timestamp)}
              </div>
            </section>
            <section className="diag-section speed-section">
              <div className="section-title">
                <Gauge size={15} /> SPEED
              </div>
              <div className="comparison-grid">
                <div>
                  <span className="metric-label pacman-color">V_EST</span>
                  <strong>{frame.vEst.toFixed(1)}</strong>
                  <small>m/s</small>
                </div>
                <div>
                  <span className="metric-label truth-color">
                    {data.meta.rfid ? 'V_REF' : 'V_TRUE'}
                  </span>
                  <strong>{frame.vTrue.toFixed(1)}</strong>
                  <small>m/s</small>
                </div>
              </div>
            </section>
            <section className="diag-section distance-section">
              <div className="section-title">
                <Route size={15} /> DISTANCE
              </div>
              <div className="distance-rows">
                <div>
                  <span>D_est</span>
                  <strong>{formatDistance(frame.dEst)}</strong>
                </div>
                <div>
                  <span>{data.meta.rfid ? 'D_ref' : 'D_true'}</span>
                  <strong>{formatDistance(frame.dTrue)}</strong>
                </div>
              </div>
              <div className="error-value negative">
                <span>
                  PACMAN ↔ {data.meta.rfid ? 'RFID REFERENCE' : 'GROUND TRUTH'}
                </span>
                <strong>{formatDistance(markerGapM)}</strong>
                <small>
                  {data.meta.rfid
                    ? 'scored only at RFID checkpoints; current cyan marker is OSM interpolation'
                    : 'straight-line distance between map markers'}
                </small>
              </div>
              {frame.display && frame.routeAlongError !== undefined && (
                <div className="distance-rows">
                  <div>
                    <span>baseline route</span>
                    <strong>
                      {frame.routeAlongError > 0 ? '+' : ''}
                      {formatDistance(frame.routeAlongError)}
                    </strong>
                  </div>
                  <div>
                    <span>saturation gate</span>
                    <strong>{frame.display.gateActive ? 'ON' : 'off'}</strong>
                  </div>
                  <div>
                    <span>ΔD (position − route)</span>
                    <strong>
                      {frame.display.deltaM > 0 ? '+' : ''}
                      {formatDistance(frame.display.deltaM)}
                    </strong>
                  </div>
                  {frame.display.excessM > 1 && (
                    <div>
                      <span>excess held at frontier</span>
                      <strong>{formatDistance(frame.display.excessM)}</strong>
                    </div>
                  )}
                </div>
              )}
              <ErrorChart
                frames={data.frames}
                index={index}
                onSeek={setPlayhead}
              />
            </section>
            <section className="diag-section edge-section">
              <div className="section-title">
                <Route size={15} /> TOPOLOGY
              </div>
              <div className="edge-row">
                <span>Tracker edge now</span>
                <code>{frame.tracker.edgeId}</code>
              </div>
              <div className="edge-row">
                <span>Truth edge now</span>
                <code>{frame.truth.edgeId}</code>
              </div>
              <div
                className={`topology-result ${frame.edgeMatch ? 'match' : 'neutral'}`}
              >
                <span className="result-dot" />
                <strong>
                  {frame.edgeMatch ? 'SAME EDGE NOW' : 'DIFFERENT EDGE NOW'}
                </strong>
              </div>
              <div
                className={`topology-result ${frame.topologyMatch ? 'match' : 'mismatch'}`}
              >
                <span className="result-dot" />
                <strong>
                  {frame.topologyMatch
                    ? 'SAME COMMITTED ROUTE'
                    : 'ROUTE CHOICE DIVERGED'}
                </strong>
              </div>
              {data.meta.rfid && (
                <p className="route-note">
                  Sparse RFID validates position at{' '}
                  {data.meta.rfid.checkpointCount} checkpoints. It cannot
                  honestly label every junction decision between them.
                </p>
              )}
              {data.meta.report.realWrongDecisionCount === 0 && (
                <p className="route-note">
                  0 real wrong turns in this replay. A different current edge
                  can mean the estimated position is behind or ahead on the
                  same route. Marker separation is {formatDistance(markerGapM)}.
                </p>
              )}
            </section>
            <section className="diag-section layer-section">
              <div className="section-title">
                <Eye size={15} /> LAYERS
              </div>
              <LayerToggle
                checked={layers.pacman}
                label="Pacman"
                accent="#f4c84d"
                onChange={toggleLayer('pacman')}
              />
              <LayerToggle
                checked={layers.trail}
                label="Pacman trail"
                accent="#f4c84d"
                onChange={toggleLayer('trail')}
              />
              <LayerToggle
                checked={layers.truth}
                label="Ground truth + trail"
                accent="#36d3c7"
                onChange={toggleLayer('truth')}
              />
              <LayerToggle
                checked={layers.basemap}
                label="OpenStreetMap"
                accent="#647178"
                onChange={toggleLayer('basemap')}
              />
              <LayerToggle
                checked={layers.diagnostics}
                label="Diagnostics"
                onChange={toggleLayer('diagnostics')}
              />
            </section>
          </aside>
        )}

        {!layers.diagnostics && (
          <Button
            className="show-diagnostics"
            variant="outline"
            onClick={() =>
              setLayers((value) => ({ ...value, diagnostics: true }))
            }
          >
            <Eye size={15} /> Diagnostics
          </Button>
        )}
      </section>
      <footer className="statusbar">
        <span>
          <i className="status-led" /> DATA READY ·{' '}
          {data.frames.length.toLocaleString()} FRAMES
        </span>
        <span>{data.meta.tripId}</span>
        <button
          type="button"
          onClick={() =>
            setLayers((value) => ({
              ...value,
              diagnostics: !value.diagnostics,
            }))
          }
        >
          {layers.diagnostics ? <EyeOff size={13} /> : <Eye size={13} />}{' '}
          {layers.diagnostics ? 'Hide diagnostics' : 'Show diagnostics'}
        </button>
      </footer>
    </main>
  );
}
