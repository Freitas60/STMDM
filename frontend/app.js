// Importar o MediaPipe diretamente via CDN ESM
import { FilesetResolver, FaceLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";

// Variáveis globais para os modelos e estado
let cnnModels = {};
let faceLandmarker = null;
let videoElement = document.getElementById('webcam');
let canvasElement = document.getElementById('output_canvas');
let canvasCtx = canvasElement.getContext('2d');

let featureWindow = [];
const WINDOW_SIZE = 150;
const FEATURE_DIM = 14;

async function initSystem() {
    const statusText = document.getElementById('system-status');
    statusText.innerText = "A carregar modelos do TensorFlow.js...";
    
    const auList = [1, 4, 6, 12, 15];
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
            document.getElementById('system-status').innerText = "Sistema a funcionar em tempo real.";
            predictLoop();
        });
    } catch (err) {
        console.error("Erro ao aceder à webcam:", err);
        document.getElementById('system-status').innerText = "Erro: Acesso à webcam negado.";
    }
}

async function predictLoop() {
    if (!faceLandmarker || videoElement.paused || videoElement.ended) return;

    let startTimeMs = performance.now();
    const results = faceLandmarker.detectForVideo(videoElement, startTimeMs);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];
        const frameFeatures = await extractFrameFeatures(landmarks);

        if (frameFeatures) {
            featureWindow.push(frameFeatures);
            if (featureWindow.length > WINDOW_SIZE) featureWindow.shift(); 
            if (featureWindow.length === WINDOW_SIZE) sendToServer([...featureWindow]);
        }
    }
    requestAnimationFrame(predictLoop);
}

async function extractFrameFeatures(landmarks) {
    return new Array(FEATURE_DIM).fill(0.0);
}

async function sendToServer(windowData) {
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ janela: windowData }),
        });
        const result = await response.json();
        document.getElementById('stress-status').innerText = `Nível de Stress: ${result.stress_level}`;
    } catch (err) {
        console.error("Erro na comunicação com o backend FastAPI:", err);
    }
}

window.onload = initSystem;
