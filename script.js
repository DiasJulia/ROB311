        const recordButton = document.getElementById('record-button');
const stopButton = document.getElementById('stop-button');
const statusText = document.getElementById('status-text');
const errorText = document.getElementById('error-text');
const playback = document.getElementById('recording-playback');

let mediaStream = null;
let mediaRecorder = null;
let audioChunks = [];
let stopTimerId = null;
let playbackUrl = null;

function setStatus(text) {
	statusText.textContent = text;
}

function setError(text) {
	errorText.textContent = text;
}

function clearError() {
	errorText.textContent = '';
}

function setControls(recording) {
	recordButton.disabled = recording;
	stopButton.disabled = !recording;
}

function cleanup() {
	if (stopTimerId) {
		clearTimeout(stopTimerId);
		stopTimerId = null;
	}

	if (mediaStream) {
		mediaStream.getTracks().forEach((track) => track.stop());
		mediaStream = null;
	}

	mediaRecorder = null;
	audioChunks = [];
}

function clearPlayback() {
	if (playbackUrl) {
		URL.revokeObjectURL(playbackUrl);
		playbackUrl = null;
	}

	playback.removeAttribute('src');
	playback.load();
}

async function startRecording() {
	clearError();
	clearPlayback();
	audioChunks = [];

	try {
		if (!navigator.mediaDevices?.getUserMedia) {
			throw new Error('getUserMedia is not available in this browser.');
		}

		mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
		mediaRecorder = new MediaRecorder(mediaStream);

		mediaRecorder.ondataavailable = (event) => {
			if (event.data && event.data.size > 0) {
				audioChunks.push(event.data);
			}
		};

		mediaRecorder.onstop = () => {
			const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
			playbackUrl = URL.createObjectURL(blob);
			playback.src = playbackUrl;
			setStatus('idle');
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
		setStatus('idle');
		setControls(false);
		setError(error instanceof Error ? error.message : 'Unable to start recording.');
	}
}

function stopRecording() {
	if (!mediaRecorder || mediaRecorder.state !== 'recording') {
		return;
	}

	setStatus('processing');
	if (stopTimerId) {
		clearTimeout(stopTimerId);
		stopTimerId = null;
	}
	mediaRecorder.stop();
}

recordButton.addEventListener('click', startRecording);
stopButton.addEventListener('click', stopRecording);
