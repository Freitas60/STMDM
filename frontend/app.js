// Importar o MediaPipe diretamente via CDN ESM
import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";

// ============================================================
// Configuração
// ============================================================
const API_URL = "http://localhost:8000/predict"; // troca pelo URL do Render quando estiver publicado

const WINDOW_SIZE = 150;      // 5 segundos a 30fps — tem de bater certo com o que a API espera
const FEATURE_DIM = 14;       // AU1/AU4/AU6 (4 classes cada) + eye_openness + hr_bpm
const CALIBRATION_SECONDS = 30;
const STRIDE_SECONDS = 1.0;

const HR_WINDOW_SEC = 6.0;
const HR_STRIDE_SEC = 1.0;

const REGION_SIZES = { testa_v1: 64, testa_v2: 96, olhos: 96 };
const AU_REGION = { 1: "testa_v1", 4: "testa_v2", 6: "olhos" };

// índices de landmarks do MediaPipe (os mesmos usados em todo o pipeline Python)
const IDX_EYE_L_OUTER = 33, IDX_EYE_L_INNER = 133;
const IDX_EYE_R_INNER = 362, IDX_EYE_R_OUTER = 263;
const IDX_FOREHEAD_REF = 10;
const IDX_EYEBROW_L_PEAK = 52, IDX_EYEBROW_R_PEAK = 282;
const IDX_EYE_L_UPPER = 159, IDX_EYE_L_LOWER = 145;
const IDX_EYE_R_UPPER = 386, IDX_EYE_R_LOWER = 374;

// ============================================================
// Estado global
// ============================================================
let cnnModels = {};
let faceLandmarker = null;
let videoElement = document.getElementById('webcam');
let canvasElement = document.getElementById('output_canvas');
let canvasCtx = canvasElement.getContext('2d', { willReadFrequently: true });

let videoFps = 30; // aproximação — o browser não expõe o fps real da webcam de forma fiável

let rgbBuffer = [];
let currentHr = null;
let lastHrUpdate = 0;

let featureWindow = [];
let lastPredictionTime = 0;

let baseline = null;          // preenchida no fim da calibração
let calibrationSamples = [];
let calibrationStart = null;
let calibrating = true;


// ============================================================
// Geometria — caixas de região e abertura do olho (iguais ao compute_region_boxes / compute_eye_openness em Python)
// Nota: aqui calculadas sobre o FRAME COMPLETO, não sobre uma cara já recortada como no
// pipeline Python (que usa um detetor de cara à parte primeiro). Como a matemática é toda
// relativa a "scale" (distância entre olhos), a geometria transfere-se bem, mas vale a
// pena confirmar visualmente que as caixas caem onde se espera antes de confiar nos números.
// ============================================================
function dist(p1, p2) {
	return Math.hypot(p1.x - p2.x, p1.y - p2.y);
}

function meanPoint(points) {
	const n = points.length;
	return {
		x: points.reduce((s, p) => s + p.x, 0) / n,
		y: points.reduce((s, p) => s + p.y, 0) / n,
	};
}

function computeRegionBoxes(landmarks, w, h) {
	const pts = landmarks.map(lm => ({ x: lm.x * w, y: lm.y * h }));
	const eyeL = meanPoint([pts[IDX_EYE_L_OUTER], pts[IDX_EYE_L_INNER]]);
	const eyeR = meanPoint([pts[IDX_EYE_R_INNER], pts[IDX_EYE_R_OUTER]]);
	const scale = dist(eyeL, eyeR);
	const eyesCenter = meanPoint([eyeL, eyeR]);
	const foreheadRef = pts[IDX_FOREHEAD_REF];
	const eyebrowsY = (pts[IDX_EYEBROW_L_PEAK].y + pts[IDX_EYEBROW_R_PEAK].y) / 2;
	const cx = eyesCenter.x;

	function rect(x0, y0, x1, y1) {
		return {
			x0: Math.max(0, Math.round(x0)), y0: Math.max(0, Math.round(y0)),
			x1: Math.min(w, Math.round(x1)), y1: Math.min(h, Math.round(y1)),
		};
	}

	return {
		testa_v1: rect(cx - 1.1 * scale, foreheadRef.y - 0.1 * scale, cx + 1.1 * scale, eyesCenter.y + 0.15 * scale),
		testa_v2: rect(cx - 1.25 * scale, foreheadRef.y - 0.25 * scale, cx + 1.25 * scale, eyebrowsY + 0.35 * scale),
		olhos: rect(cx - 1.2 * scale, eyesCenter.y - 0.45 * scale, cx + 1.2 * scale, eyesCenter.y + 0.35 * scale),
	};
}

function computeEyeOpenness(landmarks, w, h) {
	const pts = landmarks.map(lm => ({ x: lm.x * w, y: lm.y * h }));
	const eyeL = meanPoint([pts[IDX_EYE_L_OUTER], pts[IDX_EYE_L_INNER]]);
	const eyeR = meanPoint([pts[IDX_EYE_R_INNER], pts[IDX_EYE_R_OUTER]]);
	const scale = dist(eyeL, eyeR);
	const leftOpening = dist(pts[IDX_EYE_L_UPPER], pts[IDX_EYE_L_LOWER]);
	const rightOpening = dist(pts[IDX_EYE_R_UPPER], pts[IDX_EYE_R_LOWER]);
	return (leftOpening + rightOpening) / 2 / scale;
}

function getFaceBoundingBox(landmarks, w, h) {
	const xs = landmarks.map(lm => lm.x * w);
	const ys = landmarks.map(lm => lm.y * h);
	return { x0: Math.min(...xs), x1: Math.max(...xs), y0: Math.min(...ys), y1: Math.max(...ys) };
}


// ============================================================
// Extração das AUs (TF.js) + abertura do olho
// ============================================================
function cropRegionTensor(sourceCanvas, box, targetSize) {
	const tmp = document.createElement('canvas');
	tmp.width = targetSize;
	tmp.height = targetSize;
	const tmpCtx = tmp.getContext('2d');
	const bw = box.x1 - box.x0, bh = box.y1 - box.y0;
	tmpCtx.drawImage(sourceCanvas, box.x0, box.y0, bw, bh, 0, 0, targetSize, targetSize);

	return tf.tidy(() => {
		const img = tf.browser.fromPixels(tmp, 3).toFloat().div(255.0); // (targetSize, targetSize, 3)
		// luminância ponderada, igual ao cv2.cvtColor(..., COLOR_RGB2GRAY) do pipeline Python
		const weights = tf.tensor1d([0.299, 0.587, 0.114]);
		const gray = img.mul(weights).sum(-1);
		return gray.expandDims(-1).expandDims(0); // (1, targetSize, targetSize, 1)
	});
}

async function extractAuAndEyeFeatures(landmarks, w, h) {
	canvasCtx.drawImage(videoElement, 0, 0, w, h);

	const boxes = computeRegionBoxes(landmarks, w, h);
	const eyeOpenness = computeEyeOpenness(landmarks, w, h);

	const row = [];
	for (const au of [1, 4, 6]) {
		const regionName = AU_REGION[au];
		const box = boxes[regionName];
		if (box.x1 <= box.x0 || box.y1 <= box.y0) return null;
		const targetSize = REGION_SIZES[regionName];
		const inputTensor = cropRegionTensor(canvasElement, box, targetSize);
		const predTensor = cnnModels[au].predict(inputTensor);
		const predArray = await predTensor.data();
		row.push(predArray[0], predArray[1], predArray[2], predArray[3]);
		inputTensor.dispose();
		predTensor.dispose();
	}
	row.push(eyeOpenness);
	return row; // 13 valores — o hr_bpm é acrescentado à parte
}


// ============================================================
// rPPG — extração de cor (testa + bochechas), algoritmo POS, filtro, deteção de picos
// Porta o núcleo de rppg.py; simplificações assinaladas onde existem.
// ============================================================
function stdArr(arr) {
	const m = arr.reduce((s, v) => s + v, 0) / arr.length;
	const variance = arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length;
	return Math.sqrt(variance);
}

function medianArr(arr) {
	const sorted = [...arr].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

function meanColorInRegion(ctx, x0, y0, x1, y1) {
	x0 = Math.max(0, Math.round(x0)); y0 = Math.max(0, Math.round(y0));
	x1 = Math.round(x1); y1 = Math.round(y1);
	const w = x1 - x0, h = y1 - y0;
	if (w <= 0 || h <= 0) return null;
	const data = ctx.getImageData(x0, y0, w, h).data;
	let r = 0, g = 0, b = 0;
	const n = w * h;
	for (let i = 0; i < data.length; i += 4) { r += data[i]; g += data[i + 1]; b += data[i + 2]; }
	return [r / n, g / n, b / n];
}

function extractCombinedMeanRgb(ctx, landmarks, w, h) {
	const bb = getFaceBoundingBox(landmarks, w, h);
	const faceW = bb.x1 - bb.x0, faceH = bb.y1 - bb.y0;

	const forehead = meanColorInRegion(ctx, bb.x0 + 0.225 * faceW, bb.y0 + 0.05 * faceH, bb.x1 - 0.225 * faceW, bb.y0 + 0.25 * faceH);
	const leftCheek = meanColorInRegion(ctx, bb.x0 + 0.12 * faceW, bb.y0 + 0.45 * faceH, bb.x0 + 0.32 * faceW, bb.y0 + 0.68 * faceH);
	const rightCheek = meanColorInRegion(ctx, bb.x0 + 0.68 * faceW, bb.y0 + 0.45 * faceH, bb.x0 + 0.88 * faceW, bb.y0 + 0.68 * faceH);

	const regions = [forehead, leftCheek, rightCheek].filter(r => r !== null);
	if (regions.length === 0) return null;
	const n = regions.length;
	return [
		regions.reduce((s, r) => s + r[0], 0) / n,
		regions.reduce((s, r) => s + r[1], 0) / n,
		regions.reduce((s, r) => s + r[2], 0) / n,
	];
}

function posAlgorithm(rgbWindow) {
	const L = rgbWindow.length;
	const meanR = rgbWindow.reduce((s, p) => s + p[0], 0) / L;
	const meanG = rgbWindow.reduce((s, p) => s + p[1], 0) / L;
	const meanB = rgbWindow.reduce((s, p) => s + p[2], 0) / L;
	const Cn = rgbWindow.map(p => [p[0] / meanR, p[1] / meanG, p[2] / meanB]);

	const S0 = Cn.map(p => p[1] - p[2]);
	const S1 = Cn.map(p => -2 * p[0] + p[1] + p[2]);
	const alpha = stdArr(S0) / (stdArr(S1) + 1e-8);
	return S0.map((v, i) => v + alpha * S1[i]);
}

function posOverlapAdd(rgbSignal, fps, windowSec = 1.6, outlierFactor = 3.0) {
	const L = Math.round(windowSec * fps);
	const nFrames = rgbSignal.length;
	if (nFrames <= L) return null;

	const windowStds = [];
	for (let t = 0; t < nFrames - L; t++) windowStds.push(stdArr(rgbSignal.slice(t, t + L).flat()));
	const medianStd = medianArr(windowStds);

	const H = new Array(nFrames).fill(0);
	for (let t = 0; t < nFrames - L; t++) {
		const window = rgbSignal.slice(t, t + L);
		if (stdArr(window.flat()) > outlierFactor * medianStd) continue;
		let P = posAlgorithm(window);
		const m = P.reduce((s, v) => s + v, 0) / P.length;
		const sd = stdArr(P) + 1e-8;
		P = P.map(v => (v - m) / sd);
		for (let i = 0; i < L; i++) H[t + i] += P[i];
	}
	return H;
}

// Filtro Butterworth passa-banda via Fili.js (biblioteca testada, em vez de escrito à mão).
// Nota: aplicado num só sentido (Fili não tem filtfilt de fase zero) — introduz um atraso de
// fase, mas os intervalos ENTRE picos consecutivos (o que interessa para o HR) não são
// afetados por um atraso aproximadamente constante.
function bandpassFilter(signal, fps, lowHz = 0.75, highHz = 3.0, order = 3) {
	const iirCalculator = new Fili.CalcCascades();
	const coeffs = iirCalculator.bandpass({
		order: order,
		characteristic: 'butterworth',
		Fs: fps,
		Fc: (lowHz + highHz) / 2,
		BW: highHz - lowHz,
	});
	const filter = new Fili.IirFilter(coeffs);
	return signal.map(v => filter.singleStep(v));
}

// Deteção de picos simplificada (máximos locais + proeminência mínima + distância mínima).
// Não é idêntica ao scipy.signal.find_peaks do lado Python, mas funcionalmente equivalente
// para este fim.
function findPeaks(signal, minDistance, minProminence) {
	const halfWin = Math.max(1, Math.round(minDistance / 2));
	let candidates = [];
	for (let i = 1; i < signal.length - 1; i++) {
		if (signal[i] > signal[i - 1] && signal[i] >= signal[i + 1]) candidates.push(i);
	}
	candidates = candidates.filter(p => {
		const l = Math.max(0, p - halfWin), r = Math.min(signal.length, p + halfWin + 1);
		const localMin = Math.min(...signal.slice(l, r));
		return (signal[p] - localMin) >= minProminence;
	});
	candidates.sort((a, b) => signal[b] - signal[a]);
	const kept = [];
	for (const p of candidates) {
		if (kept.every(k => Math.abs(k - p) >= minDistance)) kept.push(p);
	}
	return kept.sort((a, b) => a - b);
}

function estimateHr(filteredSignal, fps, prominenceFactor = 0.8) {
	const minDistance = Math.round(fps * 60 / 180);
	const minProminence = prominenceFactor * stdArr(filteredSignal);
	const peaks = findPeaks(filteredSignal, minDistance, minProminence);
	if (peaks.length < 2) return null;
	const intervalsSec = [];
	for (let i = 1; i < peaks.length; i++) intervalsSec.push((peaks[i] - peaks[i - 1]) / fps);
	const meanIntervalSec = intervalsSec.reduce((s, v) => s + v, 0) / intervalsSec.length;
	return 60 / meanIntervalSec;
}

function updateRppgAndGetHr(landmarks, w, h) {
	const meanRgb = extractCombinedMeanRgb(canvasCtx, landmarks, w, h);
	rgbBuffer.push(meanRgb || [NaN, NaN, NaN]);
	const maxBufferLen = Math.round(HR_WINDOW_SEC * videoFps);
	if (rgbBuffer.length > maxBufferLen) rgbBuffer.shift();

	const now = performance.now() / 1000;
	if (now - lastHrUpdate > HR_STRIDE_SEC && rgbBuffer.length === maxBufferLen) {
		const clean = rgbBuffer.filter(p => !p.some(Number.isNaN));
		if (clean.length > maxBufferLen * 0.5) {
			const pulse = posOverlapAdd(clean, videoFps);
			if (pulse) {
				const filtered = bandpassFilter(pulse, videoFps);
				const hr = estimateHr(filtered, videoFps);
				if (hr !== null) currentHr = hr;
			}
		}
		lastHrUpdate = now;
	}
	return currentHr;
}


// ============================================================
// Inicialização
// ============================================================
async function initSystem() {
	const statusText = document.getElementById('system-status');
	statusText.innerText = "A carregar modelos do TensorFlow.js...";

	const auList = [1, 4, 6]; // só as 3 AUs que a LSTM lê
	for (let au of auList) {
		try {
			cnnModels[au] = await tf.loadLayersModel(`./models/au${au}_web/model.json`);
			console.log(`-> AU${au} carregada com sucesso.`);
		} catch (e) {
			console.error(`Erro ao carregar modelo AU${au}:`, e);
			statusText.innerText = `Erro ao carregar AU${au}.`;
			return;
		}
	}
	statusText.innerText = "A inicializar MediaPipe Face Landmarker...";
	try {
		const filesetResolver = await FilesetResolver.forVisionTasks(
			"https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
		);
		faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
			baseOptions: {
				modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
				delegate: "GPU"
			},
			runningMode: "VIDEO",
			numFaces: 1
		});
		statusText.innerText = "Sistema pronto. A aceder à webcam...";
		startWebcam();
	} catch (err) {
		console.error("Erro ao inicializar o MediaPipe:", err);
		statusText.innerText = "Erro ao inicializar o MediaPipe.";
	}
}

async function startWebcam() {
	try {
		const stream = await navigator.mediaDevices.getUserMedia({
			video: { width: 640, height: 480, frameRate: { ideal: 30 } }
		});
		videoElement.srcObject = stream;
		videoElement.addEventListener('loadeddata', () => {
			calibrationStart = performance.now() / 1000;
			document.getElementById('system-status').innerText =
				`CALIBRAÇÃO — fica quieto e relaxado durante ${CALIBRATION_SECONDS}s...`;
			predictLoop();
		});
	} catch (err) {
		console.error("Erro ao aceder à webcam:", err);
		document.getElementById('system-status').innerText = "Erro: Acesso à webcam negado.";
	}
}


// ============================================================
// Ciclo principal — calibração primeiro, depois janela deslizante + previsão na cloud
// ============================================================
async function predictLoop() {
	if (!faceLandmarker || videoElement.paused || videoElement.ended) return;

	const w = videoElement.videoWidth, h = videoElement.videoHeight;
	const startTimeMs = performance.now();
	const results = faceLandmarker.detectForVideo(videoElement, startTimeMs);

	if (results.faceLandmarks && results.faceLandmarks.length > 0) {
		const landmarks = results.faceLandmarks[0];
		const hr = updateRppgAndGetHr(landmarks, w, h);
		const auEyeFeatures = await extractAuAndEyeFeatures(landmarks, w, h); // 13 valores, ou null

		if (auEyeFeatures !== null) {
			const fullRow = [...auEyeFeatures, hr !== null ? hr : 0.0]; // 14 valores

			if (calibrating) {
				calibrationSamples.push(fullRow);
				const elapsed = (performance.now() / 1000) - calibrationStart;
				const restantes = Math.max(0, Math.ceil(CALIBRATION_SECONDS - elapsed));
				document.getElementById('system-status').innerText = `Calibração: ${restantes}s restantes`;
				if (elapsed >= CALIBRATION_SECONDS) {
					finishCalibration();
				}
			} else {
				const relativeRow = fullRow.map((v, i) => v - baseline[i]);
				featureWindow.push(relativeRow);
				if (featureWindow.length > WINDOW_SIZE) featureWindow.shift();

				const now = performance.now() / 1000;
				if (featureWindow.length === WINDOW_SIZE && now - lastPredictionTime > STRIDE_SECONDS) {
					lastPredictionTime = now;
					sendToServer([...featureWindow]);
				}
			}
		}
	}
	requestAnimationFrame(predictLoop);
}

function finishCalibration() {
	if (calibrationSamples.length === 0) {
		document.getElementById('system-status').innerText = "Calibração falhou — nenhuma cara detetada. Recarrega a página.";
		calibrating = true; // não avança, evita usar uma baseline vazia
		return;
	}
	const n = calibrationSamples.length;
	baseline = new Array(FEATURE_DIM).fill(0);
	for (const row of calibrationSamples) {
		for (let i = 0; i < FEATURE_DIM; i++) baseline[i] += row[i] / n;
	}
	calibrating = false;
	document.getElementById('system-status').innerText = `Baseline calculada (${n} amostras). Sistema a funcionar em tempo real.`;
	console.log("Baseline pessoal:", baseline);
}

async function sendToServer(windowData) {
	try {
		const response = await fetch(API_URL, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ janela: windowData }),
		});
		if (!response.ok) throw new Error(`HTTP ${response.status}`);
		const result = await response.json();
		document.getElementById('stress-status').innerText =
			`${result.classe.toUpperCase()} (${(result.probabilidade_stress * 100).toFixed(0)}%)`;
	} catch (err) {
		console.error("Erro na comunicação com o backend FastAPI:", err);
		document.getElementById('stress-status').innerText = "Erro a contactar a API (pode estar a acordar, até 50s)";
	}
}

window.onload = initSystem;
