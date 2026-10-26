import { HandLandmarker, FilesetResolver } from
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs";

const video = document.getElementById("video");
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const select = document.getElementById("camera-select");
const startBtn = document.getElementById("start-btn");
const mirrorToggle = document.getElementById("mirror-toggle");
const statusEl = document.getElementById("status");
const hint = document.getElementById("hint");
const handArtworks = Array.from({ length: 11 }, (_, fingers) => {
  const image = new Image();
  image.src = new URL(`./${fingers}.svg`, import.meta.url).href;
  return image;
});
const handArtworksLoaded = Promise.all(handArtworks.map((image, fingers) =>
  new Promise((resolve, reject) => {
    image.onload = resolve;
    image.onerror = () => reject(new Error(`No se pudo cargar ${fingers}.svg.`));
  }),
));

// Parámetros sintonizables vía URL:
// ?debug=1&thumbMCP=145&thumbIP=145&thumbDist=0.5
const params = new URLSearchParams(location.search);
const DEBUG = params.has("debug");
const THUMB_MCP_MIN = parseFloat(params.get("thumbMCP") || "145");
const THUMB_IP_MIN  = parseFloat(params.get("thumbIP")  || "145");
const THUMB_DIST_MIN = parseFloat(params.get("thumbDist") || "0.5");

const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
const WASM_URL =
  "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm";

// Los 4 dedos "normales": [tip, articulación PIP]
const FINGERS_DIST = [[8, 6], [12, 10], [16, 14], [20, 18]];

let handLandmarker = null;
let stream = null;
let running = false;
let rafId = null;
let lastVideoTime = -1;
let handArtworksReady = false;
let handArtworksError = null;

// ---------- Utilidades ----------

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

// Ángulo (en grados) en el punto b, formado por a-b-c
function angleDeg(a, b, c) {
  const v1x = a.x - b.x, v1y = a.y - b.y;
  const v2x = c.x - b.x, v2y = c.y - b.y;
  const dot = v1x * v2x + v1y * v2y;
  const m1 = Math.hypot(v1x, v1y);
  const m2 = Math.hypot(v2x, v2y);
  if (m1 === 0 || m2 === 0) return 0;
  const cos = Math.max(-1, Math.min(1, dot / (m1 * m2)));
  return Math.acos(cos) * 180 / Math.PI;
}

// ---------- Conteo de dedos ----------

function isThumbExtended(lm, debugInfo) {
  const palmSize = dist(lm[0], lm[9]);         // muñeca → MCP del medio
  const tipToIndex = dist(lm[4], lm[5]) / palmSize;
  const mcpAngle = angleDeg(lm[1], lm[2], lm[4]); // ángulo en la MCP del pulgar
  const ipAngle  = angleDeg(lm[2], lm[3], lm[4]); // ángulo en la IP del pulgar

  const extended =
    mcpAngle > THUMB_MCP_MIN &&
    ipAngle  > THUMB_IP_MIN  &&
    tipToIndex > THUMB_DIST_MIN;

  if (debugInfo) {
    debugInfo.mcp = Math.round(mcpAngle);
    debugInfo.ip  = Math.round(ipAngle);
    debugInfo.dst = tipToIndex.toFixed(2);
    debugInfo.ext = extended;
  }
  return extended;
}

function countFingers(lm, debugInfo) {
  const wrist = lm[0];
  let count = 0;

  // 4 dedos normales por distancia tip vs PIP a la muñeca
  for (const [tip, ref] of FINGERS_DIST) {
    if (dist(lm[tip], wrist) > dist(lm[ref], wrist) * 1.1) count++;
  }

  // Pulgar por ángulos + distancia a la MCP del índice
  if (isThumbExtended(lm, debugInfo)) count++;

  return count;
}

function drawCartoonGlove(lm) {
  const point = (index) => ({
    x: lm[index].x * canvas.width,
    y: lm[index].y * canvas.height,
  });
  const wrist = point(0);
  const middleBase = point(9);
  const palmWidth = Math.hypot(point(5).x - point(17).x, point(5).y - point(17).y);
  const outlineWidth = palmWidth * 0.065;
  const fingerWidth = palmWidth * 0.17;
  const thumbWidth = palmWidth * 0.19;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";

  const forearmX = middleBase.x - wrist.x;
  const forearmY = middleBase.y - wrist.y;
  const forearmLength = Math.hypot(forearmX, forearmY);
  if (forearmLength > 0) {
    ctx.beginPath();
    ctx.moveTo(wrist.x, wrist.y);
    ctx.lineTo(
      wrist.x - forearmX / forearmLength * Math.max(canvas.height * 0.55, palmWidth * 1.8),
      wrist.y - forearmY / forearmLength * Math.max(canvas.height * 0.55, palmWidth * 1.8),
    );
    ctx.strokeStyle = "#080808";
    ctx.lineWidth = palmWidth * 0.34;
    ctx.stroke();
  }

  const palmPoints = [0, 1, 5, 9, 13, 17].map(point);
  ctx.beginPath();
  ctx.moveTo(palmPoints[0].x, palmPoints[0].y);
  for (const p of palmPoints.slice(1)) ctx.lineTo(p.x, p.y);
  ctx.closePath();
  ctx.fillStyle = "#fff";
  ctx.fill();
  ctx.strokeStyle = "#111";
  ctx.lineWidth = outlineWidth * 2;
  ctx.stroke();

  const fingerChains = [
    { points: [1, 2, 3, 4], width: thumbWidth },
    { points: [5, 6, 7, 8], width: fingerWidth },
    { points: [9, 10, 11, 12], width: fingerWidth },
    { points: [13, 14, 15, 16], width: fingerWidth },
    { points: [17, 18, 19, 20], width: fingerWidth },
  ];

  for (const chain of fingerChains) {
    ctx.beginPath();
    chain.points.forEach((index, i) => {
      const p = point(index);
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.strokeStyle = "#111";
    ctx.lineWidth = chain.width + outlineWidth * 2;
    ctx.stroke();
    ctx.strokeStyle = "#fff";
    ctx.lineWidth = chain.width;
    ctx.stroke();
  }

  if (forearmLength > 0) {
    const outwardX = -forearmX / forearmLength;
    const outwardY = -forearmY / forearmLength;
    const cuffLength = palmWidth * 0.2;
    const cuffWidth = palmWidth * 0.44;
    const cuffCenterX = wrist.x + outwardX * cuffLength * 0.35;
    const cuffCenterY = wrist.y + outwardY * cuffLength * 0.35;

    ctx.translate(cuffCenterX, cuffCenterY);
    ctx.rotate(Math.atan2(outwardY, outwardX));
    ctx.beginPath();
    ctx.roundRect(-cuffLength / 2, -cuffWidth / 2, cuffLength, cuffWidth, cuffWidth * 0.22);
    ctx.fillStyle = "#fff";
    ctx.fill();
    ctx.strokeStyle = "#111";
    ctx.lineWidth = outlineWidth * 1.5;
    ctx.stroke();
  }

  ctx.restore();
}

function drawHandArtwork(lm, fingers, fullArtwork = false) {
  if (!handArtworksReady) {
    drawCartoonGlove(lm);
    drawFingerCount(fingers, lm[9]);
    return;
  }

  const handArtwork = handArtworks[fingers];
  const point = (index) => ({
    x: lm[index].x * canvas.width,
    y: lm[index].y * canvas.height,
  });
  const wrist = point(0);
  const middleBase = point(9);
  const palmWidth = Math.hypot(point(5).x - point(17).x, point(5).y - point(17).y);
  const downX = wrist.x - middleBase.x;
  const downY = wrist.y - middleBase.y;
  const downLength = Math.hypot(downX, downY);
  if (palmWidth === 0 || downLength === 0) return;

  const rotation = Math.atan2(downY, downX) - Math.PI / 2;
  const thumb = point(4);
  const thumbDirectionX = thumb.x - middleBase.x;
  const thumbDirectionY = thumb.y - middleBase.y;
  const rightAxisX = Math.cos(rotation);
  const rightAxisY = Math.sin(rotation);
  const flipX = thumbDirectionX * rightAxisX + thumbDirectionY * rightAxisY < 0;

  const crop = fullArtwork
    ? { x: 0, y: 0, width: 210, height: 297 }
    : { x: 45, y: 10, width: 125, height: 250 };
  const wristAnchor = { x: 105, y: 130 };
  const artworkX = flipX
    ? wristAnchor.x - crop.x - (fullArtwork ? crop.width : 0)
    : crop.x - wristAnchor.x;
  const scale = (palmWidth / 80) * 1.35;
  const sourceScaleX = handArtwork.naturalWidth / 210;
  const sourceScaleY = handArtwork.naturalHeight / 297;

  ctx.save();
  ctx.translate(wrist.x, wrist.y);
  ctx.rotate(rotation);
  ctx.scale(flipX ? -scale : scale, scale);
  ctx.drawImage(
    handArtwork,
    crop.x * sourceScaleX,
    crop.y * sourceScaleY,
    crop.width * sourceScaleX,
    crop.height * sourceScaleY,
    artworkX,
    crop.y - wristAnchor.y,
    crop.width,
    crop.height,
  );
  ctx.restore();
}

function drawFingerCount(fingers, palm) {
  const scale = Math.max(canvas.width, canvas.height) / 600;
  const px = palm.x * canvas.width;
  const py = palm.y * canvas.height;
  const fontSize = Math.max(40, 90 * scale);

  ctx.font = `bold ${fontSize}px system-ui, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.lineWidth = Math.max(4, 8 * scale);
  ctx.strokeStyle = "rgba(0,0,0,0.75)";
  ctx.strokeText(fingers, px, py);
  ctx.fillStyle = "#fbbf24";
  ctx.fillText(fingers, px, py);
}

// ---------- Modelo ----------

async function initModel() {
  setStatus("Cargando modelo de MediaPipe…");
  try {
    await handArtworksLoaded;
    handArtworksReady = true;
  } catch (error) {
    handArtworksError = error;
    console.error(error);
  }

  const vision = await FilesetResolver.forVisionTasks(WASM_URL);
  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });
  setStatus(
    handArtworksError
      ? "Modelo listo. No se pudieron cargar las poses; se mostrará el conteo numérico."
      : "Modelo listo. Elige cámara e inicia.",
  );
  startBtn.disabled = false;
}

async function refreshCameras() {
  try {
    const s = await navigator.mediaDevices.getUserMedia({ video: true });
    s.getTracks().forEach(t => t.stop());
  } catch {}
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === "videoinput");
  select.innerHTML = "";
  if (cams.length === 0) {
    select.innerHTML = '<option value="">Sin cámaras</option>';
    return;
  }
  cams.forEach((cam, i) => {
    const opt = document.createElement("option");
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Cámara ${i + 1}`;
    select.appendChild(opt);
  });
}

// ---------- Cámara ----------

async function startCamera(deviceId) {
  stopCamera();
  const constraints = {
    audio: false,
    video: deviceId ? { deviceId: { exact: deviceId } } : { facingMode: "user" },
  };
  stream = await navigator.mediaDevices.getUserMedia(constraints);
  video.srcObject = stream;
  await video.play();
  const track = stream.getVideoTracks()[0];
  setStatus(`Cámara activa: ${track.label || "desconocida"}`);
  hint.classList.add("hidden");
  if (video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  applyMirror();
  running = true;
  startBtn.textContent = "Detener";
  lastVideoTime = -1;
  rafId = requestAnimationFrame(detectLoop);
}

function stopCamera() {
  running = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId = null; }
  if (stream) {
    stream.getTracks().forEach(t => t.stop());
    stream = null;
  }
  video.srcObject = null;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  startBtn.textContent = "Iniciar";
  hint.classList.remove("hidden");
}

function applyMirror() {
  const on = mirrorToggle.checked;
  video.classList.toggle("mirror", on);
  canvas.classList.toggle("mirror", on);
}

// ---------- Detección y dibujo ----------

function detectLoop() {
  if (!running) return;

  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;

    if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    const results = handLandmarker.detectForVideo(video, performance.now());
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const hands = results.landmarks || [];
    const scale = Math.max(canvas.width, canvas.height) / 600;
    let totalFingers = 0;
    const fingerCounts = [];
    const debugInfos = [];

    for (const lm of hands) {
      const debugInfo = DEBUG ? {} : null;
      const fingers = countFingers(lm, debugInfo);
      fingerCounts.push(fingers);
      debugInfos.push(debugInfo);
      totalFingers += fingers;
    }

    const hasCombinedPose = hands.length === 2
      && totalFingers >= 6
      && totalFingers <= 10;
    if (hasCombinedPose) {
      const palmWidth = (lm) => {
        const index = lm[5];
        const pinky = lm[17];
        return Math.hypot(
          (index.x - pinky.x) * canvas.width,
          (index.y - pinky.y) * canvas.height,
        );
      };
      const mainHandIndex = palmWidth(hands[0]) >= palmWidth(hands[1]) ? 0 : 1;
      drawHandArtwork(hands[mainHandIndex], totalFingers, true);
    } else {
      hands.forEach((lm, index) => drawHandArtwork(lm, fingerCounts[index]));
    }

    for (const [index, lm] of hands.entries()) {
      const debugInfo = debugInfos[index];
      // Debug: mostrar valores del pulgar junto a la punta (landmark 4)
      if (DEBUG && debugInfo) {
        const tip = lm[4];
        const tx = tip.x * canvas.width;
        const ty = tip.y * canvas.height;
        const fsz = Math.max(12, 18 * scale);

        ctx.font = `bold ${fsz}px monospace`;
        ctx.textAlign = "left";
        ctx.textBaseline = "bottom";
        ctx.lineWidth = Math.max(2, 3 * scale);
        ctx.strokeStyle = "rgba(0,0,0,0.85)";

        const label = `MCP ${debugInfo.mcp}° IP ${debugInfo.ip}° D ${debugInfo.dst}`;
        ctx.strokeText(label, tx + 10 * scale, ty);
        ctx.fillStyle = debugInfo.ext ? "#22c55e" : "#ef4444";
        ctx.fillText(label, tx + 10 * scale, ty);
      }
    }

    // Estado inferior
    const cfg = DEBUG
      ? ` · MCP>${THUMB_MCP_MIN}° IP>${THUMB_IP_MIN}° D>${THUMB_DIST_MIN}`
      : "";
    if (hands.length === 0) {
      setStatus("Acerca tu mano a la cámara…");
    } else if (hands.length === 1) {
      setStatus(`Manos: 1 · Dedos extendidos: ${totalFingers}${cfg}`);
    } else {
      const detalle = fingerCounts.join(" + ");
      setStatus(`Manos: ${hands.length} · Dedos: ${totalFingers} (${detalle})${cfg}`);
    }
  }

  rafId = requestAnimationFrame(detectLoop);
}

// ---------- UI ----------

function setStatus(msg) { statusEl.textContent = msg; }

startBtn.addEventListener("click", async () => {
  if (running) {
    stopCamera();
    setStatus("Detenido.");
    return;
  }
  startBtn.disabled = true;
  try {
    await startCamera(select.value || undefined);
  } catch (e) {
    setStatus(`Error: ${e.message}`);
    console.error(e);
  } finally {
    startBtn.disabled = false;
  }
});

select.addEventListener("change", async () => {
  if (running) {
    try { await startCamera(select.value); }
    catch (e) { setStatus(`Error: ${e.message}`); }
  }
});

mirrorToggle.addEventListener("change", applyMirror);

// ---------- Arranque ----------

(async () => {
  try {
    await refreshCameras();
    await initModel();
  } catch (e) {
    setStatus(`Error inicial: ${e.message}`);
    console.error(e);
  }
})();
