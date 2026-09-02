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
    console.log("A carregar modelos do TensorFlow.js...");
    
    // 1. Carregar as 5 CNNs das Action Units
    const auList = [1, 4, 6, 12, 15];
    for (let au of auList) {
        try {
            cnnModels[au] = await tf.loadLayersModel(`./models/au${au}_web/model.json`);
            console.log(`-> AU${au} carregada com sucesso.`);
        } catch (e) {
            console.error(`Erro ao carregar modelo AU${au}:`, e);
        }
    }

    // 2. Inicializar MediaPipe Face Landmarker
    // (Certifica-te de incluir os scripts do MediaPipe no index.html)
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

    console.log("Sistema pronto. A iniciar captura de webcam...");
    startWebcam();
}

async function startWebcam() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
        videoElement.srcObject = stream;
        videoElement.addEventListener('loadeddata', predictLoop);
    } catch (err) {
        console.error("Erro ao aceder à webcam:", err);
    }
}

async function predictLoop() {
    if (!faceLandmarker || videoElement.paused || videoElement.ended) return;

    let startTimeMs = performance.now();
    const results = faceLandmarker.detectForVideo(videoElement, startTimeMs);

    if (results.faceLandmarks && results.faceLandmarks.length > 0) {
        const landmarks = results.faceLandmarks[0];

        // Executar processamento por frame:
        // 1. Recortar regiões das AUs e fazer inferência com as CNNs locais
        // 2. Calcular rPPG via POS e métricas de abertura ocular
        const frameFeatures = await extractFrameFeatures(landmarks);

        if (frameFeatures) {
            featureWindow.push(frameFeatures);

            // Quando atingirmos a janela de 150 frames, enviamos para o FastAPI
            if (featureWindow.length > WINDOW_SIZE) {
                featureWindow.shift(); // Remove o mais antigo (sliding window)
            }

            if (featureWindow.length === WINDOW_SIZE) {
                sendToServer(featureWindow);
            }
        }
    }

    // Continuar o loop no próximo frame
    requestAnimationFrame(predictLoop);
}

async function extractFrameFeatures(landmarks) {
    // Exemplo de vetor de 22 floats:
    // [0-4]: Outputs das 5 CNNs (AU1, AU4, AU6, AU12, AU15)
    // [5]: Abertura dos olhos
    // [6-21]: Sinais rPPG / estatísticas locais ou canais derivados
    
    let currentFeatures = new Array(FEATURE_DIM).fill(0.0);

    // TODO: Implementar crop do tensor baseado nos índices dos landmarks faciais
    // Exemplo simulado para validação do formato:
    try {
        // Exemplo: inferência fictícia da AU1 usando tensor recortado
        // const au1Input = preprocessCrop(landmarks, [indices_olho_esquerdo]);
        // const prediction = cnnModels[1].predict(au1Input);
        // currentFeatures[0] = prediction.dataSync()[0];
        
        // Simulação de preenchimento estrutural para respeitar o Pydantic [150, 22]
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

        const result = await response.json();
        console.log("Resultado de Stress (LSTM):", result);
        
        // Atualizar interface gráfica com o resultado do stress
        document.getElementById('stress-status').innerText = `Nível de Stress: ${result.stress_level}`;
    } catch (err) {
        console.error("Erro na comunicação com o backend FastAPI:", err);
    }
}

// Iniciar quando a página carregar
window.onload = initSystem;
