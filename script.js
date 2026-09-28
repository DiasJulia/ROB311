import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';

const MODEL_FILE = 'models/transfer_vgg16.onnx';
const MODEL_LABEL = 'VGG16';

const EMOTIONS = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'neutral', 'sadness'];

const SR = 16000;
const N_MELS = 128;
const N_FFT = 1024;
const HOP_LENGTH = 256;            
const WINDOW_SAMPLES = SR;         
const WINDOW_HOP = SR / 2;         
const N_FRAMES = 1 + Math.floor(WINDOW_SAMPLES / HOP_LENGTH); 
const TOP_DB = 80;                

const NORM_MEAN = 25.0;
const NORM_STD = 16.45;

const recordButton = document.getElementById('record-button');
const stopButton = document.getElementById('stop-button');
const statusText = document.getElementById('status-text');
const errorText = document.getElementById('error-text');
const playback = document.getElementById('recording-playback');
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
let microphoneStreamPromise = null;
let latestAudioBlob = null;
let sessionPromise = null;

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
	playback.removeAttribute('src');
	playback.load();
	playbackContainer.classList.add('d-none');
	inferenceContainer.classList.add('d-none');
	inferButton.disabled = true;
	inferenceText.textContent = 'Record audio first to enable inference.';
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
		const enorm = 2 / (melF[i + 2] - melF[i]); // norm='slaney'
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
		for (; right & bit; bit >>= 1) {
			right ^= bit;
		}
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
			let wReal = 1;
			let wImag = 0;

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

function melSpectrogramDb(chunk) {
	const pad = N_FFT / 2; // center=True, pad_mode='constant' (zeros)
	const padded = new Float32Array(chunk.length + 2 * pad);
	padded.set(chunk, pad);

	const out = new Float32Array(N_MELS * N_FRAMES);
	const real = new Float64Array(N_FFT);
	const imag = new Float64Array(N_FFT);
	const power = new Float64Array(N_FFT / 2 + 1);

	for (let t = 0; t < N_FRAMES; t += 1) {
		const start = t * HOP_LENGTH;
		for (let n = 0; n < N_FFT; n += 1) {
			real[n] = padded[start + n] * HANN[n];
			imag[n] = 0;
		}
		fft(real, imag);

		for (let k = 0; k < power.length; k += 1) {
			power[k] = real[k] * real[k] + imag[k] * imag[k];
		}

		for (let m = 0; m < N_MELS; m += 1) {
			const filter = MEL_FILTERS[m];
			let energy = 0;
			for (let k = 0; k < filter.length; k += 1) {
				energy += filter[k] * power[k];
			}
			out[m * N_FRAMES + t] = 10 * Math.log10(Math.max(energy, 1e-10)); // ref=1.0
		}
	}

	let max = -Infinity;
	for (let i = 0; i < out.length; i += 1) {
		if (out[i] > max) max = out[i];
	}
	const floor = max - TOP_DB;
	for (let i = 0; i < out.length; i += 1) {
		if (out[i] < floor) out[i] = floor;
	}

	return out;
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

function getSession() {
	if (!sessionPromise) {
		sessionPromise = ort.InferenceSession.create(new URL(MODEL_FILE, import.meta.url).href, {
			executionProviders: ['wasm'],
		}).catch((error) => {
			sessionPromise = null;
			throw error;
		});
	}
	return sessionPromise;
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
	if (arr.every((v) => v >= 0) && Math.abs(sum - 1) < 1e-3) {
		return arr;
	}
	const max = Math.max(...arr);
	const exps = arr.map((v) => Math.exp(v - max));
	const total = exps.reduce((a, b) => a + b, 0);
	return exps.map((v) => v / total);
}

async function runInference() {
	if (!latestAudioBlob) {
		inferenceText.textContent = 'Record audio first to run inference.';
		return;
	}

	clearError();
	inferButton.disabled = true;

	try {
		if (NORM_MEAN === null || NORM_STD === null) {
			throw new Error('Defina NORM_MEAN e NORM_STD no script.js (valores de mean e std calculados no notebook).');
		}

		inferenceText.textContent = 'Running inference...';
		const session = await getSession();
		const inputName = session.inputNames[0];
		const outputName = session.outputNames[0];
		const expectedShape = getInputShape(session, inputName);
		console.log('Input esperado pelo modelo:', expectedShape, '| enviado:', N_MELS, 'x', N_FRAMES);
		const channelsFirst = isChannelsFirst(expectedShape);
		const [expMels, expFrames] = channelsFirst ? expectedShape.slice(2) : expectedShape.slice(1, 3);
		if ((typeof expMels === 'number' && expMels !== N_MELS) || (typeof expFrames === 'number' && expFrames !== N_FRAMES)) {
			throw new Error(`Shape incompatível: o modelo espera ${expMels} bandas x ${expFrames} quadros, mas o script gera ${N_MELS} x ${N_FRAMES}. Ajuste N_MELS/HOP_LENGTH/janela para a configuração do treino.`);
		}
		const dims = channelsFirst ? [1, 1, N_MELS, N_FRAMES] : [1, N_MELS, N_FRAMES, 1];

		const samples = await blobToSamples(latestAudioBlob);
		const windows = splitWindows(samples);
		const average = new Float64Array(EMOTIONS.length);

		// média das probabilidades de todas as janelas
		for (const chunk of windows) {
			const features = melSpectrogramDb(chunk);
			for (let i = 0; i < features.length; i += 1) {
				features[i] = (features[i] - NORM_MEAN) / NORM_STD;
			}

			const outputs = await session.run({ [inputName]: new ort.Tensor('float32', features, dims) });
			const probs = toProbabilities(outputs[outputName].data);
			for (let i = 0; i < EMOTIONS.length; i += 1) {
				average[i] += probs[i] / windows.length;
			}
		}

		const ranking = Array.from(average, (p, i) => ({ label: EMOTIONS[i], p })).sort((a, b) => b.p - a.p);
		const fmt = (r) => `${r.label} ${(r.p * 100).toFixed(1)}%`;
		inferenceText.textContent = `${MODEL_LABEL}: ${fmt(ranking[0])} (${ranking.slice(1, 3).map(fmt).join(', ')})`;
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
		if (!navigator.mediaDevices?.getUserMedia) {
			throw new Error('getUserMedia is not available in this browser.');
		}

		if (!mediaStream) {
			if (!microphoneStreamPromise) {
				microphoneStreamPromise = navigator.mediaDevices.getUserMedia({
					audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false },
				});
			}

			mediaStream = await microphoneStreamPromise;
		}

		mediaRecorder = new MediaRecorder(mediaStream);

		mediaRecorder.ondataavailable = (event) => {
			if (event.data && event.data.size > 0) {
				audioChunks.push(event.data);
			}
		};

		mediaRecorder.onstop = () => {
			latestAudioBlob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
			playbackUrl = URL.createObjectURL(latestAudioBlob);
			playback.src = playbackUrl;
			playback.load();
			playbackContainer.classList.remove('d-none');
			inferenceContainer.classList.remove('d-none');
			inferenceText.textContent = 'Ready. Press "Run inference".';
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
		microphoneStreamPromise = null;
	}
}

function stopRecording() {
	if (!mediaRecorder || (mediaRecorder.state !== 'recording' && mediaRecorder.state !== 'paused')) {
		return;
	}

	setStatus('processing');
	if (stopTimerId) {
		clearTimeout(stopTimerId);
		stopTimerId = null;
	}
	mediaRecorder.stop();
}

window.addEventListener('beforeunload', () => {
	if (mediaStream) {
		mediaStream.getTracks().forEach((track) => track.stop());
		mediaStream = null;
	}

	if (microphoneStreamPromise) {
		microphoneStreamPromise = null;
	}
});

function toggleRecording() {
	if (!mediaRecorder) {
		void startRecording();
		return;
	}

	if (mediaRecorder.state === 'recording') {
		mediaRecorder.pause();
		setStatus('paused');
		setControls(true);
		return;
	}

	if (mediaRecorder.state === 'paused') {
		mediaRecorder.resume();
		setStatus('recording');
		setControls(true);
	}
}

const option = document.createElement('option');
option.value = 'vgg16';
option.textContent = MODEL_LABEL;
modelSelect.replaceChildren(option);

recordButton.addEventListener('click', toggleRecording);
if (stopButton) {
	stopButton.addEventListener('click', stopRecording);
}
inferButton.addEventListener('click', () => {
	void runInference();
});