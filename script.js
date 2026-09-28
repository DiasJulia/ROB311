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
let microphoneStreamPromise = null;

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

		if (!mediaStream) {
			if (!microphoneStreamPromise) {
				microphoneStreamPromise = navigator.mediaDevices.getUserMedia({ audio: true });
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
			const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || 'audio/webm' });
			playbackUrl = URL.createObjectURL(blob);
			playback.src = playbackUrl;
			playback.load();
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

recordButton.addEventListener('click', toggleRecording);
if (stopButton) {
	stopButton.addEventListener('click', stopRecording);
}
