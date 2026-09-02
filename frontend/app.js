// ==========================================================
// CONFIGURAÇÕES GLOBAIS
// ==========================================================
const CLOUD_API_URL = "http://localhost:8000/predict";
const FPS = 30;
const WINDOW_SIZE = 150; 
const N_FEATURES = 22;

let featureBuffer = [];
let rgbBuffer = []; 
let currentBPM = 0.0;

// Estado da Calibração T1
let isCalibrating = false;
let isCalibrated = false;
let calibrationBuffer = [];
let baselineT1 = new Array(N_FEATURES).fill(0.0);

// Modelos
let faceLandmarker;
let modelsAU = {};

// DOM
const videoEl = document.getElementById('webcam');
const canvasEl = document.getElementById('output-canvas');
const ctx = canvasEl.getContext('2d');
const statusText = document.getElementById('status-text');
const predictionText = document.getElementById('prediction-text');
const btnCalibrate = document.getElementById('btn-calibrate');

// ==========================================================
// 1. INICIALIZAÇÃO DE MODELOS E CÂMARA
// ==========================================================
async function initSystem() {
    statusText.innerText = "A carregar Modelos (MediaPipe e CNNs)...";
    
    // 1.1 Carregar MediaPipe FaceLandmarker
    const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm");
    faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
        },
        runningMode: "VIDEO",
        numFaces: 1
    });

    // 1.2 Carregar 5 CNNs via TensorFlow.js
    const auList = [1, 4, 6, 12, 15];
    for (let au of auList) {
        try {
            modelsAU[au] = await tf.loadLayersModel(`models/au${au}_web/model.json`);
        } catch (e) {
            console.warn(`Aviso: Modelo AU${au} não encontrado. Simulando dados...`);
        }
    }

    statusText.innerText = "Modelos Prontos. A ligar câmara...";

    // 1.3 Iniciar Webcam
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480, facingMode: "user" } });
    videoEl.srcObject = stream;
    
    videoEl.addEventListener('loadeddata', () => {
        canvasEl.width = videoEl.videoWidth;
        canvasEl.height = videoEl.videoHeight;
        btnCalibrate.disabled = false;
        statusText.innerText = "Sistema Pronto.";
        requestAnimationFrame(frameLoop);
    });
}

// ==========================================================
// 2. MATEMÁTICA E BIOMETRIA
// ==========================================================
function calculateEAR(landmarks, w, h) {
    const p = (idx) => ({ x: landmarks[idx].x * w, y: landmarks[idx].y * h });
    const dist = (p1, p2) => Math.sqrt(Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2));
    
    const vDist = dist(p(159), p(145)); // Topo-Fundo
    const hDist = dist(p(33), p(133));  // Esquerda-Direita
    return hDist > 0 ? (vDist / hDist) : 0.0;
}

function getForeheadMeanRGB(landmarks, ctx, w, h) {
    const cx = Math.floor(landmarks[10].x * w);
    const cy = Math.floor(landmarks[10].y * h);
    const boxSize = 20; 
    
    // Evitar sair dos limites do canvas
    if (cx < boxSize || cy < boxSize || cx > w-boxSize || cy > h-boxSize) return [0,0,0];

    const imgData = ctx.getImageData(cx - boxSize/2, cy - boxSize/2, boxSize, boxSize).data;
    let r = 0, g = 0, b = 0;
    for (let i = 0; i < imgData.length; i += 4) {
        r += imgData[i];
        g += imgData[i+1];
        b += imgData[i+2];
    }
    const totalPixels = boxSize * boxSize;
    return [r / totalPixels, g / totalPixels, b / totalPixels];
}

function runPOSAlgorithm(buffer) {
    const N = buffer.length;
    const meanR = buffer.reduce((a, v) => a + v[0], 0) / N;
    const meanG = buffer.reduce((a, v) => a + v[1], 0) / N;
    const meanB = buffer.reduce((a, v) => a + v[2], 0) / N;

    let X = new Float32Array(N);
    let Y = new Float32Array(N);

    for (let i = 0; i < N; i++) {
        const Rn = buffer[i][0] / meanR;
        const Gn = buffer[i][1] / meanG;
        const Bn = buffer[i][2] / meanB;
        X[i] = Gn - Bn;
        Y[i] = -2 * Rn + Gn + Bn;
    }

    const meanX = X.reduce((a, b) => a + b, 0) / N;
    const meanY = Y.reduce((a, b) => a + b, 0) / N;
    const stdX = Math.sqrt(X.reduce((a, v) => a + Math.pow(v - meanX, 2), 0) / N);
    const stdY = Math.sqrt(Y.reduce((a, v) => a + Math.pow(v - meanY, 2), 0) / N);
    
    const alpha = stdX / (stdY + 1e-8);
    let pulse = new Float32Array(N);
    for (let i = 0; i < N; i++) pulse[i] = X[i] + alpha * Y[i];
    return pulse;
}

function estimateBPM(pulseSignal, fps) {
    let peaks = [];
    const minDistance = Math.floor(fps * (60 / 180)); 
    
    for (let i = 1; i < pulseSignal.length - 1; i++) {
        if (pulseSignal[i] > pulseSignal[i - 1] && pulseSignal[i] > pulseSignal[i + 1]) {
            if (peaks.length === 0 || (i - peaks[peaks.length - 1]) >= minDistance) {
                peaks.push(i);
            }
        }
    }
    if (peaks.length < 2) return currentBPM; // Retorna último valor válido

    let totalIntervals = 0;
    for (let i = 1; i < peaks.length; i++) {
        totalIntervals += (peaks[i] - peaks[i - 1]);
    }
    const meanIntervalSecs = (totalIntervals / (peaks.length - 1)) / fps;
    return 60.0 / meanIntervalSecs;
}

// ==========================================================
// 3. EVENTOS DA INTERFACE (CALIBRAÇÃO T1)
// ==========================================================
btnCalibrate.addEventListener('click', () => {
    isCalibrating = true;
    isCalibrated = false;
    calibrationBuffer = [];
    statusText.innerText = "A gravar Baseline T1... Por favor, não se mova.";
    btnCalibrate.disabled = true;
});

// ==========================================================
// 4. LOOP PRINCIPAL DE PROCESSAMENTO
// ==========================================================
let lastVideoTime = -1;

async function frameLoop() {
    ctx.drawImage(videoEl, 0, 0, canvasEl.width, canvasEl.height);
    
    let startTimeMs = performance.now();
    let results = null;
    
    if (videoEl.currentTime !== lastVideoTime) {
        lastVideoTime = videoEl.currentTime;
        results = faceLandmarker.detectForVideo(videoEl, startTimeMs);
    }

    if (results && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        let currentFrameFeatures = new Array(N_FEATURES).fill(0.0);
        
        // PARTE 1: CNNs (Mockup estrutural até implementares recortes via TF.js)
        let idx = 0;
        const auList = [1, 4, 6, 12, 15];
        for (let au of auList) {
            // Logica futura: tf.browser.fromPixels(crop), rgb2gray, model.predict()
            for (let c = 0; c < 4; c++) {
                currentFrameFeatures[idx++] = Math.random(); // Dummy
            }
        }

        // PARTE 2 e 3: EAR e rPPG
        const currentEAR = calculateEAR(landmarks, canvasEl.width, canvasEl.height);
        currentFrameFeatures[20] = currentEAR;
        document.getElementById('val-ear').innerText = currentEAR.toFixed(2);

        const meanRGB = getForeheadMeanRGB(landmarks, ctx, canvasEl.width, canvasEl.height);
        rgbBuffer.push(meanRGB);
        if (rgbBuffer.length > WINDOW_SIZE) rgbBuffer.shift();
        
        if (rgbBuffer.length === WINDOW_SIZE) {
            const rawPulse = runPOSAlgorithm(rgbBuffer);
            currentBPM = estimateBPM(rawPulse, FPS);
            document.getElementById('val-bpm').innerText = currentBPM.toFixed(0);
        }
        currentFrameFeatures[21] = currentBPM;

        // PARTE 4: Lógica de Calibração / Normalização
        if (isCalibrating) {
            calibrationBuffer.push(currentFrameFeatures);
            if (calibrationBuffer.length === WINDOW_SIZE) {
                // Calcular média para cada feature
                for (let i = 0; i < N_FEATURES; i++) {
                    baselineT1[i] = calibrationBuffer.reduce((sum, frame) => sum + frame[i], 0) / WINDOW_SIZE;
                }
                isCalibrating = false;
                isCalibrated = true;
                statusText.innerText = "Calibração Completa. Sistema Ativo.";
                btnCalibrate.innerText = "Recalibrar T1";
                btnCalibrate.disabled = false;
            }
        } 
        else if (isCalibrated) {
            // Normalizar subtraindo a baseline
            let normalizedFeatures = currentFrameFeatures.map((val, i) => val - baselineT1[i]);
            featureBuffer.push(normalizedFeatures);
            
            // Submeter janela para a nuvem e avançar
            if (featureBuffer.length === WINDOW_SIZE) {
                sendToCloud(featureBuffer);
                featureBuffer = featureBuffer.slice(75); // Stride = 2.5 segs
            }
        }
    }
    
    requestAnimationFrame(frameLoop);
}

// ==========================================================
// 5. COMUNICAÇÃO COM A CLOUD
// ==========================================================
async function sendToCloud(matrix) {
    try {
        const response = await fetch(CLOUD_API_URL, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ janela: matrix })
        });
        const result = await response.json();
        
        if (result.probabilidade_stress !== undefined) {
            predictionText.innerText = `Classe: ${result.classe.toUpperCase()} (${(result.probabilidade_stress*100).toFixed(1)}%)`;
            predictionText.style.color = result.classe === "stress" ? "#ff4444" : "#00ffcc";
        }
    } catch (err) {
        console.error("Erro na API Cloud:", err);
    }
}

// Iniciar a aplicação
initSystem();
