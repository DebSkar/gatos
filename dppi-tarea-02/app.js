import {
  HandLandmarker,
  FaceLandmarker,
  FilesetResolver,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

// ---- Meme Mapping (Gatitoactualizado) ---------------------------------
const GESTURE_MEMES = {
  default: ["Gatitoactualizado/7.Duda.jpeg"],
  orejaGato: ["Gatitoactualizado/1.Oreja de gato.jpeg"],
  mirarAbajo: ["Gatitoactualizado/2.MirarHaciaAbajo.jpeg"],
  lengua: ["Gatitoactualizado/3.Lengua.jpeg"],
  cabezaAtras: ["Gatitoactualizado/4.CabezaHaciaAtras.jpeg"],
  brazosAtras: ["Gatitoactualizado/5.Brazos hacia atras.jpeg"],
  shocked: ["Gatitoactualizado/6.Shocked.jpeg"],
  duda: ["Gatitoactualizado/7.Duda.jpeg"],
  pistola: ["Gatitoactualizado/8.Pistola.jpeg"],
};

const GESTURE_NAMES = {
  default: "En reposo (Haz un gesto)",
  orejaGato: "1. Orejas de Gato",
  mirarAbajo: "2. Mirar Hacia Abajo",
  lengua: "3. Sacar la Lengua",
  cabezaAtras: "4. Cabeza Hacia Atrás",
  brazosAtras: "5. Brazos Hacia Atrás",
  shocked: "6. Brazos Cruzados (Shocked)",
  duda: "7. Cara de Duda",
  pistola: "8. Pistola",
};

const STABLE_FRAMES_REQUIRED = 4;
const DEFAULT_FALLBACK_MS = 600;
const FACE_STALE_MS = 1200;

const video = document.getElementById("video");
const memeImg = document.getElementById("memeImg");
const debugHud = document.getElementById("debugHud");

let handLandmarker, faceLandmarker;
let lastVideoTime = -1;
let currentGesture = "default";
let candidateGesture = "default";
let candidateStreak = 0;
let lastNonDefaultAt = performance.now();
let lastFace = null;
let lastYawDebug = 0;
let lastPitchDebug = 0;
let lastMouthOpenDebug = 0;

async function init() {
  if (debugHud) debugHud.textContent = "1/3 Descargando motor MediaPipe WASM...";

  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
  );

  if (debugHud) debugHud.textContent = "2/3 Cargando modelos de Inteligencia Artificial...";

  // 1. Cargar HandLandmarker con fallback a CPU si GPU falla
  try {
    handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
  } catch (errGpu) {
    console.warn("GPU delegate no soportado para HandLandmarker, usando CPU...", errGpu);
    handLandmarker = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numHands: 2,
    });
  }

  // 2. Cargar FaceLandmarker con fallback a CPU
  try {
    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "GPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
    });
  } catch (errGpu) {
    console.warn("GPU delegate no soportado para FaceLandmarker, usando CPU...", errGpu);
    faceLandmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
        delegate: "CPU",
      },
      runningMode: "VIDEO",
      numFaces: 1,
      outputFacialTransformationMatrixes: true,
    });
  }

  if (debugHud) debugHud.textContent = "3/3 Solicitando acceso a la cámara web...";

  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;

  await new Promise((resolve) => {
    video.onloadedmetadata = () => {
      video.play().then(resolve).catch(resolve);
    };
  });

  if (debugHud) debugHud.textContent = "¡Listo! Detectando gestos en tiempo real...";
  requestAnimationFrame(loop);
}

// ---- Geometry Helpers --------------------------------------------------
function vec(a, b) {
  return { x: b.x - a.x, y: b.y - a.y, z: (b.z || 0) - (a.z || 0) };
}

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y, (a.z || 0) - (b.z || 0));
}

function angleDeg(v1, v2) {
  const dot = v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
  const m1 = Math.hypot(v1.x, v1.y, v1.z);
  const m2 = Math.hypot(v2.x, v2.y, v2.z);
  if (m1 < 1e-9 || m2 < 1e-9) return 180;
  return (Math.acos(Math.min(1, Math.max(-1, dot / (m1 * m2)))) * 180) / Math.PI;
}

function fingerExtended(lm, mcp, pip, tip) {
  const angle = angleDeg(vec(lm[mcp], lm[pip]), vec(lm[pip], lm[tip]));
  return angle < 45;
}

function headPoseAngles(matrixObj) {
  if (!matrixObj) return { yaw: 0, pitch: 0 };
  const data = matrixObj.data || matrixObj;
  if (!data || data.length < 16) return { yaw: 0, pitch: 0 };

  const r00 = data[0], r10 = data[4], r20 = data[8];
  const r21 = data[9], r22 = data[10], r11 = data[5], r12 = data[6];

  const sy = Math.hypot(r00, r10);
  let yaw = 0, pitch = 0;
  if (sy < 1e-6) {
    yaw = 0;
    pitch = (Math.atan2(-r12, r11) * 180) / Math.PI;
  } else {
    yaw = (Math.atan2(-r20, sy) * 180) / Math.PI;
    pitch = (Math.atan2(r21, r22) * 180) / Math.PI;
  }
  return { yaw, pitch };
}

function classifyHand(lm) {
  if (!lm || lm.length < 21) return null;
  const handScale = dist(lm[0], lm[9]) || 1e-6;

  const indexUp = fingerExtended(lm, 5, 6, 8);
  const middleUp = fingerExtended(lm, 9, 10, 12);
  const ringUp = fingerExtended(lm, 13, 14, 16);
  const pinkyUp = fingerExtended(lm, 17, 18, 20);

  const thumbPinkySpread = dist(lm[4], lm[17]) / handScale;
  const thumbOut = thumbPinkySpread > 1.0;

  const curledCount = [indexUp, middleUp, ringUp, pinkyUp].filter((v) => !v).length;

  return {
    indexUp,
    middleUp,
    ringUp,
    pinkyUp,
    thumbOut,
    curledCount,
    handScale,
    indexTip: lm[8],
    middleTip: lm[12],
    ringTip: lm[16],
    pinkyTip: lm[20],
    thumbTip: lm[4],
    indexMCP: lm[5],
    middleMCP: lm[9],
    wrist: lm[0],
    palmCenter: lm[9],
    pts: lm,
  };
}

function isPointing(h) {
  return h && h.indexUp && !h.middleUp && !h.ringUp && !h.pinkyUp;
}

function updateFace(faceResult) {
  const now = performance.now();
  const sawFace = !!(faceResult && faceResult.faceLandmarks && faceResult.faceLandmarks.length > 0);

  if (sawFace) {
    const f = faceResult.faceLandmarks[0];
    const upperLip = f[13];
    const lowerLip = f[14];
    const rightCheek = f[234];
    const leftCheek = f[454];
    const noseTip = f[1];
    const forehead = f[10];
    const chin = f[152];

    const mouthCenter = {
      x: (upperLip.x + lowerLip.x) / 2,
      y: (upperLip.y + lowerLip.y) / 2,
      z: ((upperLip.z || 0) + (lowerLip.z || 0)) / 2,
    };
    const faceWidth = dist(rightCheek, leftCheek) || 1e-6;
    const mouthOpen = dist(upperLip, lowerLip) / faceWidth;

    const noseForeheadDist = Math.abs(noseTip.y - forehead.y) + 1e-6;
    const chinNoseDist = Math.abs(chin.y - noseTip.y);
    const verticalRatio = chinNoseDist / noseForeheadDist;

    let yaw = 0, pitch = 0;
    if (faceResult.facialTransformationMatrixes && faceResult.facialTransformationMatrixes.length > 0) {
      const angles = headPoseAngles(faceResult.facialTransformationMatrixes[0]);
      yaw = angles.yaw;
      pitch = angles.pitch;
    }

    lastFace = {
      mouthCenter,
      faceWidth,
      mouthOpen,
      yawDeg: yaw,
      pitchDeg: pitch,
      verticalRatio,
      rightCheek,
      leftCheek,
      t: now,
    };
    lastYawDebug = yaw;
    lastPitchDebug = pitch;
    lastMouthOpenDebug = mouthOpen;
  }
}

function decideGesture(handResult) {
  const now = performance.now();
  const faceIsFresh = !!lastFace && now - lastFace.t < FACE_STALE_MS;
  const rawHands = (handResult && handResult.landmarks && handResult.landmarks.length > 0)
    ? handResult.landmarks.map(classifyHand).filter(Boolean)
    : [];

  // --- 1. GESTOS DE 2 MANOS ---
  if (rawHands.length >= 2) {
    const [h1, h2] = rawHands;

    // Gesto 1: orejaGato (índice y medio extendidos en ambas manos)
    const bothEars = (h1.indexUp && h1.middleUp && !h1.ringUp && !h1.pinkyUp) &&
                     (h2.indexUp && h2.middleUp && !h2.ringUp && !h2.pinkyUp);
    if (bothEars) {
      let bothHigh = false;
      if (faceIsFresh) {
        const headTop = lastFace.mouthCenter.y - lastFace.faceWidth * 0.5;
        bothHigh = h1.palmCenter.y < headTop || h1.wrist.y < headTop;
      } else {
        bothHigh = h1.palmCenter.y < 0.45 && h2.palmCenter.y < 0.45;
      }
      if (bothHigh) return "orejaGato";
    }

    // Gesto 5: brazosAtras (manos elevadas y separadas lateralmente)
    const handsDistX = Math.abs(h1.wrist.x - h2.wrist.x);
    const bothElevated = h1.wrist.y < 0.8 && h2.wrist.y < 0.8;
    if (bothElevated && handsDistX > 0.40) {
      return "brazosAtras";
    }

    // Gesto 6: shocked (brazos cruzados en el pecho)
    let bothChest = false;
    if (faceIsFresh) {
      const chestY = lastFace.mouthCenter.y + lastFace.faceWidth * 0.15;
      bothChest = h1.wrist.y > chestY && h2.wrist.y > chestY;
    } else {
      bothChest = h1.wrist.y > 0.40 && h2.wrist.y > 0.40;
    }
    const wristsCloseX = Math.abs(h1.wrist.x - h2.wrist.x) < 0.40;
    const wristsCloseY = Math.abs(h1.wrist.y - h2.wrist.y) < 0.35;
    if (bothChest && wristsCloseX && wristsCloseY) {
      return "shocked";
    }
  }

  // --- 2. GESTOS DE 1 MANO ---
  for (const h of rawHands) {
    // Gesto 7: duda (índice en la sien / lateral de la cabeza)
    if (isPointing(h) && faceIsFresh) {
      const dR = dist(h.indexTip, lastFace.rightCheek) / lastFace.faceWidth;
      const dL = dist(h.indexTip, lastFace.leftCheek) / lastFace.faceWidth;
      const nearTemple = (dR < 0.65 || dL < 0.65) && (h.indexTip.y < lastFace.mouthCenter.y - lastFace.faceWidth * 0.1);
      if (nearTemple) return "duda";
    }

    // Gesto 8: pistola (índice y medio juntos hacia arriba, pulgar afuera)
    const isGun = h.indexUp && h.middleUp && !h.ringUp && !h.pinkyUp && h.thumbOut;
    if (isGun) {
      const tipGap = dist(h.indexTip, h.middleTip) / h.handScale;
      if (tipGap < 0.55) return "pistola";
    }
  }

  // --- 3. GESTOS FACIALES ---
  if (faceIsFresh) {
    // Gesto 3: lengua
    if (lastFace.mouthOpen > 0.22) {
      return "lengua";
    }

    // Gesto 4: cabezaAtras
    if (lastFace.pitchDeg > 11.0 || lastFace.verticalRatio > 1.45) {
      return "cabezaAtras";
    }

    // Gesto 2: mirarAbajo
    if (lastFace.pitchDeg < -11.0 || lastFace.verticalRatio < 0.50) {
      return "mirarAbajo";
    }
  }

  return "default";
}

function pickImage(gesture) {
  const images = GESTURE_MEMES[gesture] || GESTURE_MEMES.default;
  const rawPath = images[Math.floor(Math.random() * images.length)];
  return encodeURI(rawPath);
}

function applyGesture(gesture) {
  if (gesture === currentGesture) return;
  currentGesture = gesture;
  memeImg.src = pickImage(gesture);
}

function loop() {
  const now = performance.now();
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime && video.videoWidth > 0) {
    lastVideoTime = video.currentTime;
    const ts = performance.now();

    try {
      const handResult = handLandmarker ? handLandmarker.detectForVideo(video, ts) : null;
      const faceResult = faceLandmarker ? faceLandmarker.detectForVideo(video, ts) : null;
      
      if (faceResult) updateFace(faceResult);
      const gesture = decideGesture(handResult);

      if (gesture === candidateGesture) {
        candidateStreak++;
      } else {
        candidateGesture = gesture;
        candidateStreak = 1;
      }

      if (candidateStreak >= STABLE_FRAMES_REQUIRED) {
        applyGesture(gesture);
      }

      if (gesture !== "default") lastNonDefaultAt = now;
      if (now - lastNonDefaultAt > DEFAULT_FALLBACK_MS && currentGesture !== "default") {
        applyGesture("default");
      }

      updateDebugHud();
    } catch (loopErr) {
      console.warn("Error en frame de detección:", loopErr);
    }
  }
  requestAnimationFrame(loop);
}

function updateDebugHud() {
  if (!debugHud) return;
  const nombreGesto = GESTURE_NAMES[currentGesture] || currentGesture;
  debugHud.textContent =
    `--- GATITO ACTUALIZADO (8 GESTOS) ---\n` +
    `Gesto: ${nombreGesto}\n` +
    `Pitch: ${lastPitchDebug >= 0 ? "+" : ""}${lastPitchDebug.toFixed(1)}° (atrás >+11, abajo <-11)\n` +
    `Boca: ${lastMouthOpenDebug.toFixed(2)} (lengua >0.22)\n` +
    `Yaw: ${lastYawDebug >= 0 ? "+" : ""}${lastYawDebug.toFixed(1)}°`;
}

init().catch((err) => {
  console.error("Error al inicializar:", err);
  if (debugHud) {
    debugHud.style.color = "#ff6b6b";
    debugHud.textContent =
      `⚠️ Error de inicio:\n${err.message || err}\n\n` +
      `Si es un error de cámara, autoriza el permiso de cámara en tu navegador.\n` +
      `Asegúrate de abrir http://localhost:8000 en Chrome, Safari o Firefox.`;
  }
});


