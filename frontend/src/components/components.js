import * as Tone from "tone";

const SCORE_NOTES = {
  severe: ["C2", "Eb2", "Gb2", "Bb2"],
  moderate: ["C3", "F3", "G3", "Bb3"],
  healthy: ["C4", "E4", "G4", "B4"],
};

const MELODY_SCALES = {
  severe: ["C3", "Eb3", "F3", "Gb3", "Bb3"],
  moderate: ["C4", "D4", "F4", "G4", "Bb4"],
  healthy: ["C5", "E5", "G5", "A5", "B5"],
};

function scoreBand(score) {
  if (score < 34) return "severe";
  if (score < 67) return "moderate";
  return "healthy";
}

function lerp(a, b, t) {
  return a + (b - a) * Math.max(0, Math.min(1, t));
}

let initialized = false;
let enabled = false;
let enablePromise = null;

let drone;
let noiseSource;
let noiseFilter;
let melodySynth;
let droneGain;
let noiseGain;
let melodyGain;
let masterGain;
let melodyLoop;
let lastScore = 50;
let lastBand = "moderate";

function buildDrone() {
  droneGain = new Tone.Gain(0).connect(masterGain);

  drone = new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: "sine" },
    envelope: { attack: 4, decay: 1, sustain: 1, release: 6 },
    volume: -18,
  }).connect(droneGain);

  const reverb = new Tone.Reverb({ decay: 8, wet: 0.7 });
  drone.connect(reverb);
  reverb.connect(droneGain);
}

function buildNoise() {
  noiseGain = new Tone.Gain(0).connect(masterGain);

  noiseFilter = new Tone.Filter({
    type: "bandpass",
    frequency: 800,
    Q: 0.8,
  });

  noiseSource = new Tone.Noise("brown").connect(noiseFilter);
  noiseFilter.connect(new Tone.Gain(0.08)).connect(noiseGain);
  noiseSource.start();
}

function buildMelody() {
  melodyGain = new Tone.Gain(0).connect(masterGain);

  melodySynth = new Tone.Synth({
    oscillator: { type: "triangle" },
    envelope: { attack: 0.8, decay: 0.4, sustain: 0.2, release: 2.5 },
    volume: -22,
  });

  const delay = new Tone.FeedbackDelay({ delayTime: "8n", feedback: 0.3, wet: 0.4 });
  const reverb = new Tone.Reverb({ decay: 5, wet: 0.5 });

  melodySynth.chain(delay, reverb, melodyGain);
}

function pickMelodyNote(score) {
  const band = scoreBand(score);
  const scale = MELODY_SCALES[band];
  return scale[Math.floor(Math.random() * scale.length)];
}

function melodyInterval(score) {
  if (score < 34) return "2n";
  if (score < 67) return "2n.";
  return "1n";
}

function applyScore(score) {
  const band = scoreBand(score);
  const now = Tone.now();

  drone.releaseAll();
  drone.triggerAttack(SCORE_NOTES[band], now + 0.1);

  noiseFilter.frequency.rampTo(lerp(200, 1800, score / 100), 2);
  noiseGain.gain.rampTo(lerp(0.05, 0.22, 1 - score / 100), 2);

  if (band !== lastBand && melodyLoop) {
    melodyLoop.interval = melodyInterval(score);
  }

  lastBand = band;
  lastScore = score;
}

async function init() {
  if (initialized) return;

  await Tone.start();
  Tone.getTransport().bpm.value = 60;

  masterGain = new Tone.Gain(0.85).toDestination();

  buildDrone();
  buildNoise();
  buildMelody();

  initialized = true;
}

export async function enable() {
  if (enabled) return;
  if (enablePromise) return enablePromise;

  enablePromise = (async () => {
    await init();

    enabled = true;

    droneGain.gain.rampTo(0.6, 2);
    noiseGain.gain.rampTo(0.12, 2);
    melodyGain.gain.rampTo(0.5, 2);

    if (!melodyLoop) {
      melodyLoop = new Tone.Loop((time) => {
        if (!enabled) return;
        const note = pickMelodyNote(lastScore);
        melodySynth.triggerAttackRelease(note, "4n", time);
      }, melodyInterval(lastScore));
    }

    melodyLoop.start(0);
    Tone.getTransport().start();
    applyScore(lastScore);
  })();

  try {
    await enablePromise;
  } finally {
    enablePromise = null;
  }
}

export function disable() {
  enabled = false;

  if (melodyLoop) {
    melodyLoop.stop();
    melodyLoop.dispose();
    melodyLoop = null;
  }

  Tone.getTransport().stop();

  droneGain?.gain.rampTo(0, 1.5);
  noiseGain?.gain.rampTo(0, 1.5);
  melodyGain?.gain.rampTo(0, 1.5);

  drone?.releaseAll();
}

export function updateScore(score) {
  lastScore = score;
  if (enabled && initialized) {
    applyScore(score);
  }
}

export function toggle() {
  if (enabled) {
    disable();
  } else {
    void enable();
  }
  return !enabled;
}

export function isEnabled() {
  return enabled;
}
