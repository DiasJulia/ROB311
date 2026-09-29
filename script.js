import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs';
import WaveSurfer from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.esm.js';
import Spectrogram from 'https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/plugins/spectrogram.esm.js';

import config from './config.json' with { type: 'json' };

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

const EMOTIONS = config.emotions;
const MODELS = config.models;

const SR = 16000;
const N_MELS = 128;
const N_FFT = 1024;
const WINDOW_SAMPLES = SR;         
const WINDOW_HOP = SR / 2;         
const TOP_DB = 80;                

const NORM_MEAN = 25.0;
const NORM_STD = 16.45;

const recordButton = document.getElementById('record-button');
const stopButton = document.getElementById('stop-button');
const statusText = document.getElementById('status-text');
const errorText = document.getElementById('error-text');
const playButton = document.getElementById('play-button');
const timeText = document.getElementById('time-text');
const spectrogramWrap = document.getElementById('spectrogram-wrap');
const spectrogramCursor = document.getElementById('spectrogram-cursor');
const playbackContainer = document.getElementById('playback-container');
const inferenceContainer = document.getElementById('inference-container');
const modelSelect = document.getElementById('model-select');
const inferButton = document.getElementById('infer-button');
const inferenceText = document.getElementById('inference-text');

let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let stopTimerId = null;
let playbackUrl = null;
let latestAudioBlob = null;

// Cache de sessões ONNX
const sessionPromises = new Map();

let audioContext = null;
let analyserNode = null;
let wavesurfer = null;

function setStatus(text) {
	statusText.textContent = text;
}

function setError(text) {
	errorText.textContent = text;
	errorText.classList.remove('d-none');
}

function clearError() {
	errorText.textContent = '';
	errorText.classList.add('d-none');
}

function setControls(recording) {
	if (stopButton) {
		stopButton.disabled = !recording;
	}
	recordButton.classList.toggle('is-recording', recording);
	recordButton.setAttribute('aria-label', recording ? 'Pause recording' : 'Start recording');
	recordButton.setAttribute('title', recording ? 'Pause recording' : 'Start recording');
}

function cleanup() {
	if (stopTimerId) {
		clearTimeout(stopTimerId);
		stopTimerId = null;
	}
	mediaRecorder = null;
	audioChunks = [];
}

function clearPlayback() {
	if (playbackUrl) {
		URL.revokeObjectURL(playbackUrl);
		playbackUrl = null;
	}

	latestAudioBlob = null;
	playbackContainer.classList.add('d-none');
	inferenceContainer.classList.add('d-none');
	inferButton.disabled = true;
	inferenceText.textContent = 'Record audio first to enable inference.';

	if (wavesurfer) {
		wavesurfer.destroy();
		wavesurfer = null;
	}
	playButton.disabled = true;
	playButton.textContent = '▶ Play';
	updateProgress(0, 0);
}

function formatTime(seconds) {
	const m = Math.floor(seconds / 60);
	const s = Math.floor(seconds % 60);
	return `${m}:${String(s).padStart(2, '0')}`;
}

function updateProgress(current, total) {
	const pct = total > 0 ? Math.min(current / total, 1) * 100 : 0;
	spectrogramCursor.style.left = `${pct}%`;
	timeText.textContent = `${formatTime(current)} / ${formatTime(total)}`;
}

function setupWavesurfer(url) {
	wavesurfer = WaveSurfer.create({
		container: '#waveform',
		waveColor: '#0d6efd',
		progressColor: '#0a58ca',
		cursorColor: '#dc3545',
		cursorWidth: 2,
		height: 80,
		plugins: [
			Spectrogram.create({
				container: '#spectrogram',
				labels: true,
				height: 120,
				splitChannels: false
			})
		]
	});

	wavesurfer.on('ready', (duration) => {
		playButton.disabled = false;
		updateProgress(0, duration);
	});
	wavesurfer.on('timeupdate', (t) => updateProgress(t, wavesurfer.getDuration()));
	wavesurfer.on('play', () => { playButton.textContent = '⏸ Pause'; });
	wavesurfer.on('pause', () => { playButton.textContent = '▶ Play'; });

	wavesurfer.load(url);
}

const F_SP = 200 / 3;
const MIN_LOG_HZ = 1000;
const MIN_LOG_MEL = MIN_LOG_HZ / F_SP;
const LOG_STEP = Math.log(6.4) / 27;
const hzToMel = (f) => (f >= MIN_LOG_HZ ? MIN_LOG_MEL + Math.log(f / MIN_LOG_HZ) / LOG_STEP : f / F_SP);
const melToHz = (m) => (m >= MIN_LOG_MEL ? MIN_LOG_HZ * Math.exp(LOG_STEP * (m - MIN_LOG_MEL)) : F_SP * m);

function buildMelFilterbank() {
	const nBins = N_FFT / 2 + 1;
	const melMin = hzToMel(0);
	const melMax = hzToMel(SR / 2);
	const melF = Array.from({ length: N_MELS + 2 }, (_, i) => melToHz(melMin + ((melMax - melMin) * i) / (N_MELS + 1)));

	const filters = [];
	for (let i = 0; i < N_MELS; i += 1) {
		const filter = new Float32Array(nBins);
		const enorm = 2 / (melF[i + 2] - melF[i]);
		const dLow = melF[i + 1] - melF[i];
		const dHigh = melF[i + 2] - melF[i + 1];

		for (let k = 0; k < nBins; k += 1) {
			const freq = (k * SR) / N_FFT;
			const lower = (freq - melF[i]) / dLow;
			const upper = (melF[i + 2] - freq) / dHigh;
			filter[k] = Math.max(0, Math.min(lower, upper)) * enorm;
		}
		filters.push(filter);
	}
	return filters;
}

const HANN = Float64Array.from({ length: N_FFT }, (_, n) => 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / N_FFT));
const MEL_FILTERS = buildMelFilterbank();

function fft(real, imag) {
	const size = real.length;
	for (let left = 1, right = 0; left < size; left += 1) {
		let bit = size >> 1;
		for (; right & bit; bit >>= 1) right ^= bit;
		right ^= bit;
		if (left < right) {
			[real[left], real[right]] = [real[right], real[left]];
			[imag[left], imag[right]] = [imag[right], imag[left]];
		}
	}
	for (let length = 2; length <= size; length <<= 1) {
		const half = length >> 1;
		const angle = (-2 * Math.PI) / length;
		const stepReal = Math.cos(angle);
		const stepImag = Math.sin(angle);
		for (let start = 0; start < size; start += length) {
			let wReal = 1, wImag = 0;
			for (let offset = 0; offset < half; offset += 1) {
				const even = start + offset;
				const odd = even + half;
				const oddReal = wReal * real[odd] - wImag * imag[odd];
				const oddImag = wReal * imag[odd] + wImag * real[odd];
				real[odd] = real[even] - oddReal;
				imag[odd] = imag[even] - oddImag;
				real[even] += oddReal;
				imag[even] += oddImag;
				const nextReal = wReal * stepReal - wImag * stepImag;
				wImag = wReal * stepImag + wImag * stepReal;
				wReal = nextReal;
			}
		}
	}
}

function melSpectrogramDb(chunk, hopLength) {
	const nFrames = 1 + Math.floor(WINDOW_SAMPLES / hopLength);
	const pad = N_FFT / 2;
	const padded = new Float32Array(chunk.length + 2 * pad);
	padded.set(chunk, pad);
	const out = new Float32Array(N_MELS * nFrames);
	const real = new Float64Array(N_FFT);
	const imag = new Float64Array(N_FFT);
	const power = new Float64Array(N_FFT / 2 + 1);

	for (let t = 0; t < nFrames; t += 1) {
		const start = t * hopLength;
		for (let n = 0; n < N_FFT; n += 1) {
			real[n] = padded[start + n] * HANN[n];
			imag[n] = 0;
		}
		fft(real, imag);
		for (let k = 0; k < power.length; k += 1) power[k] = real[k] * real[k] + imag[k] * imag[k];
		for (let m = 0; m < N_MELS; m += 1) {
			const filter = MEL_FILTERS[m];
			let energy = 0;
			for (let k = 0; k < filter.length; k += 1) energy += filter[k] * power[k];
			out[m * nFrames + t] = 10 * Math.log10(Math.max(energy, 1e-10));
		}
	}
	let max = -Infinity;
	for (let i = 0; i < out.length; i += 1) if (out[i] > max) max = out[i];
	const floor = max - TOP_DB;
	for (let i = 0; i < out.length; i += 1) if (out[i] < floor) out[i] = floor;
	return { features: out, nFrames };
}

function splitWindows(samples) {
	if (samples.length <= WINDOW_SAMPLES) {
		const single = new Float32Array(WINDOW_SAMPLES);
		single.set(samples);
		return [single];
	}
	const windows = [];
	for (let s = 0; s + WINDOW_SAMPLES <= samples.length; s += WINDOW_HOP) {
		windows.push(samples.subarray(s, s + WINDOW_SAMPLES));
	}
	return windows;
}

async function blobToSamples(blob) {
	const context = new AudioContext();
	const decoded = await context.decodeAudioData(await blob.arrayBuffer());
	await context.close();
	const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * SR), SR);
	const source = offline.createBufferSource();
	source.buffer = decoded;
	source.connect(offline.destination);
	source.start();
	const resampled = await offline.startRendering();
	return new Float32Array(resampled.getChannelData(0));
}

function getSession(filePath) {
	if (!sessionPromises.has(filePath)) {
		const promise = ort.InferenceSession.create(new URL(filePath, import.meta.url).href, {
			executionProviders: ['wasm'],
		}).catch((error) => {
			sessionPromises.delete(filePath);
			throw error;
		});
		sessionPromises.set(filePath, promise);
	}
	return sessionPromises.get(filePath);
}

function getInputShape(session, inputName) {
	const meta = session.inputMetadata;
	const info = Array.isArray(meta) ? meta.find((m) => m.name === inputName) : meta?.[inputName];
	return info?.shape ?? info?.dimensions ?? [];
}

function isChannelsFirst(shape) {
	return shape.length === 4 && shape[1] === 1 && shape[3] !== 1;
}

function toProbabilities(values) {
	const arr = Array.from(values);
	const sum = arr.reduce((a, b) => a + b, 0);
	if (arr.every((v) => v >= 0) && Math.abs(sum - 1) < 1e-3) return arr;
	const max = Math.max(...arr);
	const exps = arr.map((v) => Math.exp(v - max));
	const total = exps.reduce((a, b) => a + b, 0);
	return exps.map((v) => v / total);
}

async function runInference() {
	if (!latestAudioBlob) { inferenceText.textContent = 'Record audio first to run inference.'; return; }
	
	const selectedKey = modelSelect?.value;
	const selectedModel = MODELS[selectedKey];

	if (!selectedModel) {
		setError('Please select a valid model.');
		return;
	}

	clearError();
	inferButton.disabled = true;

	try {
		inferenceText.textContent = `Loading ${selectedModel.label} & running inference...`;
		
		const session = await getSession(selectedModel.file);
		const inputName = session.inputNames[0];
		const outputName = session.outputNames[0];
		const expectedShape = getInputShape(session, inputName);
		const channelsFirst = isChannelsFirst(expectedShape);

		const hopLength = selectedModel.hop_length || 256;

		const samples = await blobToSamples(latestAudioBlob);
		const windows = splitWindows(samples);
		const average = new Float64Array(EMOTIONS.length);

		for (const chunk of windows) {
			const { features, nFrames } = melSpectrogramDb(chunk, hopLength);
			for (let i = 0; i < features.length; i += 1) {
				features[i] = (features[i] - NORM_MEAN) / NORM_STD;
			}
			
			const dims = channelsFirst ? [1, 1, N_MELS, nFrames] : [1, N_MELS, nFrames, 1];
			const outputs = await session.run({ [inputName]: new ort.Tensor('float32', features, dims) });
			const probs = toProbabilities(outputs[outputName].data);
			for (let i = 0; i < EMOTIONS.length; i += 1) average[i] += probs[i] / windows.length;
		}

		const ranking = Array.from(average, (p, i) => ({ label: EMOTIONS[i], p })).sort((a, b) => b.p - a.p);
		const fmt = (r) => `${r.label} ${(r.p * 100).toFixed(1)}%`;
		inferenceText.textContent = `${selectedModel.label}: ${fmt(ranking[0])} (${ranking.slice(1, 3).map(fmt).join(', ')})`;
	} catch (error) {
		setError(error instanceof Error ? error.message : 'Inference failed.');
		inferenceText.textContent = 'Inference failed.';
	} finally {
		inferButton.disabled = false;
	}
}

async function startRecording() {
	clearError();
	clearPlayback();
	audioChunks = [];

	try {
		if (!navigator.mediaDevices?.getUserMedia) throw new Error('getUserMedia is not available.');

		if (!mediaStream) {
			mediaStream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
		}

		audioContext = new (window.AudioContext || window.webkitAudioContext)();
		const sourceNode = audioContext.createMediaStreamSource(mediaStream);
		analyserNode = audioContext.createAnalyser();
		analyserNode.fftSize = 256;
		sourceNode.connect(analyserNode);

		mediaRecorder = new MediaRecorder(mediaStream);
		mediaRecorder.ondataavailable = (event) => { if (event.data && event.data.size > 0) audioChunks.push(event.data); };

		mediaRecorder.onstop = () => {
			latestAudioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
			playbackUrl = URL.createObjectURL(latestAudioBlob);

			playbackContainer.classList.remove('d-none');
			inferenceContainer.classList.remove('d-none');
			setupWavesurfer(playbackUrl);

			inferenceText.textContent = 'Ready. Select a model and press "Evaluate".';
			inferButton.disabled = false;
			setStatus('Press to start recording');
			setControls(false);
			cleanup();
		};

		mediaRecorder.start();
		setStatus('recording');
		setControls(true);

		stopTimerId = window.setTimeout(() => {
			stopRecording();
		}, 3000);
	} catch (error) {
		cleanup();
		setStatus('Press to start recording');
		setControls(false);
		setError(error instanceof Error ? error.message : 'Unable to start recording.');
	}
}

function stopRecording() {
	if (!mediaRecorder || (mediaRecorder.state !== 'recording' && mediaRecorder.state !== 'paused')) return;
	setStatus('processing');
	if (stopTimerId) { clearTimeout(stopTimerId); stopTimerId = null; }
	mediaRecorder.stop();
}

window.addEventListener('beforeunload', () => {
	if (mediaStream) { mediaStream.getTracks().forEach((track) => track.stop()); mediaStream = null; }
});

function toggleRecording() {
	if (!mediaRecorder) return void startRecording();
	if (mediaRecorder.state === 'recording') { mediaRecorder.pause(); setStatus('paused'); setControls(true); return; }
	if (mediaRecorder.state === 'paused') { mediaRecorder.resume(); setStatus('recording'); setControls(true); }
}

for (const [key, model] of Object.entries(MODELS)) {
    if (modelSelect) {
        const option = document.createElement('option');
        option.value = key;
        option.textContent = model.label;
        modelSelect.appendChild(option);
    }
}

recordButton.addEventListener('click', toggleRecording);
if (stopButton) stopButton.addEventListener('click', stopRecording);
inferButton.addEventListener('click', () => { void runInference(); });

playButton.addEventListener('click', () => wavesurfer?.playPause());

spectrogramWrap.addEventListener('click', (e) => {
	if (!wavesurfer) return;
	const rect = spectrogramWrap.getBoundingClientRect();
	wavesurfer.seekTo((e.clientX - rect.left) / rect.width);
});