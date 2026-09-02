// Variáveis globais para os modelos e estado
let cnnModels = {};
let faceLandmarker = null;
let videoElement = document.getElementById('webcam');
let canvasElement = document.getElementById('output_canvas');
let canvasCtx = canvasElement.getContext('2d');

// Buffer para a janela temporal de 150 frames (5 segundos a 30fps)
let featureWindow = [];
const WINDOW_SIZE = 150;
const FEATURE_DIM = 22;

async function initSystem() {
    const statusText = document.getElementById('system-status');
    statusText.innerText = "A carregar modelos do TensorFlow.js...";
    
    // 1. Carregar as 5 CNNs das Action Units a partir da pasta models/
    const auList = [1, 4, 6, 12, 15];
    for (let au of auList) {
        try {
            cnnModels[au] = await tf.loadLayersModel(`./models/au${au}_web/model.json`);
            console.log(`-> AU${au} carregada com sucesso.`);
        } catch (e) {
            console.error(`Erro ao carregar modelo AU${au}:`, e);
            statusText.innerText = `Erro ao carregar AU${au}. Verifica a consola.`;
            return;
        }
    }

    statusText.innerText = "A inicializar MediaPipe Face Landmarker...";

    // 2. Inicializar MediaPipe Face Landmarker
    try {
        const vision = await FilesetResolver.forVisionTasks(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
        );
        
        faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
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
        document.getElementById('system-status').innerText = "Erro: Acesso à webcam negado ou indisponível.";
    }
}

async function predictLoop() {
    if (!faceLandmarker || videoElement.paused || videoElement.ended) return;

    let startTimeMs = performance.now();
    const results = faceLandmarker.detectForVideo(videoElement, startTimeMs);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];

        // Extrair features do frame atual (CNNs edge + rPPG)
        const frameFeatures = await extractFrameFeatures(landmarks);

        if (frameFeatures) {
            featureWindow.push(frameFeatures);

            // Janela deslizante de 150 frames
            if (featureWindow.length > WINDOW_SIZE) {
                featureWindow.shift(); 
            }

            // Quando preenchermos a janela de 5 segundos, enviamos para o FastAPI
            if (featureWindow.length === WINDOW_SIZE) {
                sendToServer([...featureWindow]); // Envia cópia da janela atual
            }
        }
    }

    // Continuar o loop no próximo frame do browser
    requestAnimationFrame(predictLoop);
}

async function extractFrameFeatures(landmarks) {
    let currentFeatures = new Array(FEATURE_DIM).fill(0.0);

    try {
        // NOTA: Podes desenhar o frame atual no canvas auxiliar para fazer crops se necessário:
        // canvasCtx.drawImage(videoElement, 0, 0, canvasElement.width, canvasElement.height);

        // Exemplo de inferência para as CNNs locais (0 a 4):
        // Para cada AU, podes recortar a bounding box respetiva dos landmarks, converter para tensor 
        // e passar pelo modelo correspondente:
        // const tensorAU1 = preprocessCropAndToTensor(canvasElement, landmarks, [indices_relevantes]);
        // const pred1 = cnnModels[1].predict(tensorAU1);
        // currentFeatures[0] = pred1.dataSync()[0];
        // tensorAU1.dispose();

        // Para já, preenchemos a estrutura base (podes expandir com os teus cálculos reais de rPPG e AUs):
        return currentFeatures;
    } catch (err) {
        console.error("Erro na extração de features do frame:", err);
        return null;
    }
}

async function sendToServer(windowData) {
    try {
        const response = await fetch('http://localhost:8000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ window: windowData })
        });

        if (!response.ok) {
            throw new Error(`Erro HTTP: ${response.status}`);
        }

        const result = await response.json();
        console.log("Resultado de Stress (LSTM):", result);
        
        // Atualizar interface gráfica com o resultado retornado pelo FastAPI
        const stressElem = document.getElementById('stress-status');
        stressElem.innerText = `Nível de Stress: ${result.stress_level} (Score:${result.score?.toFixed(2) || 'N/A'})`;
    } catch (err) {
        console.error("Erro na comunicação com o backend FastAPI:", err);
    }
}

// Iniciar o sistema automaticamente quando a página carregar
window.onload = initSystem;
