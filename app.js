import {
  FilesetResolver,
  ImageEmbedder,
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/+esm";

const statusEl = document.getElementById("status");
const resultsPanel = document.getElementById("resultsPanel");

const video = document.getElementById("video");
const overlay = document.getElementById("overlay");
const overlayCtx = overlay.getContext("2d");

const uploadedPreview = document.getElementById("uploadedPreview");
const uploadInput = document.getElementById("uploadInput");
const runImageBtn = document.getElementById("runImageBtn");

const frameCanvas = document.getElementById("frameCanvas");
const frameCtx = frameCanvas.getContext("2d", { willReadFrequently: true });

const grayCanvas = document.getElementById("grayCanvas");
const grayCtx = grayCanvas.getContext("2d", { willReadFrequently: true });

const cropCanvas = document.getElementById("cropCanvas");
const cropCtx = cropCanvas.getContext("2d", { willReadFrequently: true });

const rotCanvas = document.getElementById("rotCanvas");
const rotCtx = rotCanvas.getContext("2d", { willReadFrequently: true });

const referenceImage = document.getElementById("referenceImage");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");

const thresholdInput = document.getElementById("threshold");
const maxProposalsInput = document.getElementById("maxProposals");
const minBoxSizeInput = document.getElementById("minBoxSize");
const frameStrideInput = document.getElementById("frameStride");

const thresholdValue = document.getElementById("thresholdValue");
const maxProposalsValue = document.getElementById("maxProposalsValue");
const minBoxSizeValue = document.getElementById("minBoxSizeValue");
const frameStrideValue = document.getElementById("frameStrideValue");

let imageEmbedder = null;
let refEmbeddings = [];
let webcamStream = null;
let animationFrameId = null;
let uploadedImageObj = null;
let isRunningWebcam = false;
let frameCount = 0;

const ROTATIONS = [0, 45, 90, 135, 180, 225, 270, 315];
const REFERENCE_SCALES = [1.0, 1.25, 1.5];
const PROPOSAL_STRIDE = 8;

function setStatus(text) {
  statusEl.textContent = text;
}

function updateSliderLabels() {
  thresholdValue.textContent = Number(thresholdInput.value).toFixed(2);
  maxProposalsValue.textContent = maxProposalsInput.value;
  minBoxSizeValue.textContent = minBoxSizeInput.value;
  frameStrideValue.textContent = frameStrideInput.value;
}

thresholdInput.addEventListener("input", updateSliderLabels);
maxProposalsInput.addEventListener("input", updateSliderLabels);
minBoxSizeInput.addEventListener("input", updateSliderLabels);
frameStrideInput.addEventListener("input", updateSliderLabels);
updateSliderLabels();

async function initMediaPipe() {
  setStatus("Loading MediaPipe...");
  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.22/wasm",
  );

  imageEmbedder = await ImageEmbedder.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/image_embedder/mobilenet_v3_small/float32/latest/mobilenet_v3_small.tflite",
    },
    runningMode: "IMAGE",
    l2Normalize: true,
    quantize: false,
  });

  await buildReferenceEmbeddings();
  setStatus("Ready");
  resultsPanel.innerHTML = `<div class="small">Model loaded. Start webcam or upload an image.</div>`;
}

async function buildReferenceEmbeddings() {
  await waitForImage(referenceImage);

  refEmbeddings = [];

  for (const scale of REFERENCE_SCALES) {
    for (const angle of ROTATIONS) {
      const canvas = makeTransformedReference(referenceImage, angle, scale);
      const embedding = embedCanvas(canvas);
      if (embedding) {
        refEmbeddings.push({
          angle,
          scale,
          vector: embedding,
        });
      }
    }
  }
}

function waitForImage(img) {
  return new Promise((resolve, reject) => {
    if (img.complete && img.naturalWidth > 0) {
      resolve();
      return;
    }

    img.onload = () => resolve();
    img.onerror = reject;
  });
}

function makeTransformedReference(img, angleDeg, scale = 1) {
  const baseW = img.naturalWidth;
  const baseH = img.naturalHeight;
  const side = Math.ceil(Math.max(baseW, baseH) * scale * 1.8);

  rotCanvas.width = side;
  rotCanvas.height = side;
  rotCtx.clearRect(0, 0, side, side);

  rotCtx.save();
  rotCtx.translate(side / 2, side / 2);
  rotCtx.rotate((angleDeg * Math.PI) / 180);
  rotCtx.scale(scale, scale);
  rotCtx.drawImage(img, -baseW / 2, -baseH / 2);
  rotCtx.restore();

  const out = document.createElement("canvas");
  out.width = side;
  out.height = side;
  out.getContext("2d").drawImage(rotCanvas, 0, 0);

  return out;
}

function embedCanvas(canvas) {
  try {
    const result = imageEmbedder.embed(canvas);
    return result.embeddings?.[0]?.floatEmbedding || null;
  } catch (error) {
    console.error("Embed failed:", error);
    return null;
  }
}

function cosineSimilarity(a, b) {
  let dot = 0;
  let normA = 0;
  let normB = 0;

  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }

  return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-8);
}

function bestReferenceSimilarity(candidateEmbedding) {
  let best = -1;
  let bestMeta = null;

  for (const ref of refEmbeddings) {
    const score = cosineSimilarity(candidateEmbedding, ref.vector);
    if (score > best) {
      best = score;
      bestMeta = ref;
    }
  }

  return { score: best, ref: bestMeta };
}

function prepareCanvasFromSource(source, width, height) {
  frameCanvas.width = width;
  frameCanvas.height = height;
  grayCanvas.width = width;
  grayCanvas.height = height;

  frameCtx.clearRect(0, 0, width, height);
  frameCtx.drawImage(source, 0, 0, width, height);

  const imageData = frameCtx.getImageData(0, 0, width, height);
  const gray = toGray(imageData);
  grayCtx.putImageData(gray, 0, 0);

  return { imageData, gray };
}

function toGray(imageData) {
  const data = imageData.data;
  const out = new ImageData(imageData.width, imageData.height);

  for (let i = 0; i < data.length; i += 4) {
    const g = Math.round(
      0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2],
    );
    out.data[i] = g;
    out.data[i + 1] = g;
    out.data[i + 2] = g;
    out.data[i + 3] = 255;
  }

  return out;
}

function sobelMagnitude(grayImageData) {
  const { width, height, data } = grayImageData;
  const mags = new Float32Array(width * height);

  const get = (x, y) => data[(y * width + x) * 4];

  for (let y = 1; y < height - 1; y++) {
    for (let x = 1; x < width - 1; x++) {
      const gx =
        -1 * get(x - 1, y - 1) +
        1 * get(x + 1, y - 1) +
        -2 * get(x - 1, y) +
        2 * get(x + 1, y) +
        -1 * get(x - 1, y + 1) +
        1 * get(x + 1, y + 1);

      const gy =
        -1 * get(x - 1, y - 1) +
        -2 * get(x, y - 1) +
        -1 * get(x + 1, y - 1) +
        1 * get(x - 1, y + 1) +
        2 * get(x, y + 1) +
        1 * get(x + 1, y + 1);

      mags[y * width + x] = Math.sqrt(gx * gx + gy * gy);
    }
  }

  return mags;
}

function buildIntegralMap(values, width, height) {
  const integral = new Float32Array((width + 1) * (height + 1));

  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    for (let x = 1; x <= width; x++) {
      rowSum += values[(y - 1) * width + (x - 1)];
      integral[y * (width + 1) + x] =
        integral[(y - 1) * (width + 1) + x] + rowSum;
    }
  }

  return integral;
}

function rectSum(integral, width, x, y, w, h) {
  const stride = width + 1;
  const x1 = x;
  const y1 = y;
  const x2 = x + w;
  const y2 = y + h;

  return (
    integral[y2 * stride + x2] -
    integral[y1 * stride + x2] -
    integral[y2 * stride + x1] +
    integral[y1 * stride + x1]
  );
}

function generateProposals(width, height, grayImageData) {
  const minBox = Number(minBoxSizeInput.value);
  const maxProposals = Number(maxProposalsInput.value);

  const mags = sobelMagnitude(grayImageData);
  const integral = buildIntegralMap(mags, width, height);

  const proposalSizes = [];
  const shortSide = Math.min(width, height);
  for (
    let size = minBox;
    size <= shortSide * 0.5;
    size = Math.round(size * 1.35)
  ) {
    proposalSizes.push(size);
  }

  const proposals = [];

  for (const size of proposalSizes) {
    for (let y = 0; y <= height - size; y += PROPOSAL_STRIDE) {
      for (let x = 0; x <= width - size; x += PROPOSAL_STRIDE) {
        const score =
          rectSum(integral, width, x, y, size, size) / (size * size);
        proposals.push({ x, y, w: size, h: size, proposalScore: score });
      }
    }
  }

  proposals.sort((a, b) => b.proposalScore - a.proposalScore);
  return proposals.slice(0, maxProposals);
}

function cropBoxToCanvas(sourceCanvas, box, padding = 0.15) {
  const padX = Math.round(box.w * padding);
  const padY = Math.round(box.h * padding);

  const sx = Math.max(0, box.x - padX);
  const sy = Math.max(0, box.y - padY);
  const sw = Math.min(sourceCanvas.width - sx, box.w + padX * 2);
  const sh = Math.min(sourceCanvas.height - sy, box.h + padY * 2);

  const size = 224;
  cropCanvas.width = size;
  cropCanvas.height = size;
  cropCtx.clearRect(0, 0, size, size);
  cropCtx.drawImage(sourceCanvas, sx, sy, sw, sh, 0, 0, size, size);

  return {
    canvas: cropCanvas,
    sourceRect: { x: sx, y: sy, w: sw, h: sh },
  };
}

function iou(a, b) {
  const x1 = Math.max(a.x, b.x);
  const y1 = Math.max(a.y, b.y);
  const x2 = Math.min(a.x + a.w, b.x + b.w);
  const y2 = Math.min(a.y + a.h, b.y + b.h);

  const interW = Math.max(0, x2 - x1);
  const interH = Math.max(0, y2 - y1);
  const interArea = interW * interH;

  const union = a.w * a.h + b.w * b.h - interArea;
  return union <= 0 ? 0 : interArea / union;
}

function nonMaxSuppression(boxes, threshold = 0.35) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const keep = [];

  while (sorted.length) {
    const current = sorted.shift();
    keep.push(current);

    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(current, sorted[i]) > threshold) {
        sorted.splice(i, 1);
      }
    }
  }

  return keep;
}

function drawDetections(detections, width, height) {
  overlay.width = width;
  overlay.height = height;
  overlayCtx.clearRect(0, 0, width, height);

  overlayCtx.lineWidth = 3;
  overlayCtx.font = "16px sans-serif";

  detections.forEach((det, index) => {
    overlayCtx.strokeStyle = "rgba(255, 216, 77, 0.95)";
    overlayCtx.fillStyle = "rgba(255, 216, 77, 0.18)";
    overlayCtx.fillRect(det.x, det.y, det.w, det.h);
    overlayCtx.strokeRect(det.x, det.y, det.w, det.h);

    const label = `star ${(det.score * 100).toFixed(1)}%`;
    const pad = 6;
    const metrics = overlayCtx.measureText(label);
    const boxW = metrics.width + pad * 2;
    const boxH = 24;
    const lx = det.x;
    const ly = Math.max(0, det.y - boxH);

    overlayCtx.fillStyle = "rgba(0, 0, 0, 0.78)";
    overlayCtx.fillRect(lx, ly, boxW, boxH);
    overlayCtx.fillStyle = "#ffd84d";
    overlayCtx.fillText(label, lx + pad, ly + 17);
  });
}

function renderResults(detections, elapsedMs, proposalCount) {
  if (!detections.length) {
    resultsPanel.innerHTML = `
      <div class="result-miss">
        <strong>No star detected.</strong><br />
        Checked ${proposalCount} proposals in ${elapsedMs.toFixed(1)} ms.
      </div>
      <div class="small">
        Try lowering the similarity threshold, moving the object closer, or using a cleaner reference image.
      </div>
    `;
    return;
  }

  resultsPanel.innerHTML = `
    <div class="result-hit">
      <strong>${detections.length} detection${detections.length === 1 ? "" : "s"}</strong><br />
      Checked ${proposalCount} proposals in ${elapsedMs.toFixed(1)} ms.
    </div>
    ${detections
      .map(
        (d, i) => `
          <div class="small">
            #${i + 1} — score ${(d.score * 100).toFixed(1)}% at
            x:${Math.round(d.x)}, y:${Math.round(d.y)},
            w:${Math.round(d.w)}, h:${Math.round(d.h)}
          </div>
        `,
      )
      .join("")}
  `;
}

function detectFromCurrentFrame() {
  if (!imageEmbedder) return;

  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) return;

  const started = performance.now();
  prepareCanvasFromSource(video, width, height);

  const proposals = generateProposals(
    width,
    height,
    grayCtx.getImageData(0, 0, width, height),
  );
  const threshold = Number(thresholdInput.value);

  const hits = [];

  for (const proposal of proposals) {
    const { canvas, sourceRect } = cropBoxToCanvas(frameCanvas, proposal, 0.15);
    const embedding = embedCanvas(canvas);
    if (!embedding) continue;

    const match = bestReferenceSimilarity(embedding);
    if (match.score >= threshold) {
      hits.push({
        x: sourceRect.x,
        y: sourceRect.y,
        w: sourceRect.w,
        h: sourceRect.h,
        score: match.score,
      });
    }
  }

  const finalDetections = nonMaxSuppression(hits, 0.3);
  drawDetections(finalDetections, width, height);
  renderResults(finalDetections, performance.now() - started, proposals.length);
}

function animationLoop() {
  if (!isRunningWebcam) return;

  frameCount++;
  const stride = Number(frameStrideInput.value);

  if (frameCount % stride === 0) {
    detectFromCurrentFrame();
  }

  animationFrameId = requestAnimationFrame(animationLoop);
}

async function startWebcam() {
  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "environment",
        width: { ideal: 1280 },
        height: { ideal: 720 },
      },
      audio: false,
    });

    uploadedPreview.style.display = "none";
    video.style.display = "block";
    video.srcObject = webcamStream;

    await video.play();

    isRunningWebcam = true;
    startBtn.disabled = true;
    stopBtn.disabled = false;
    setStatus("Webcam running");
    animationLoop();
  } catch (error) {
    console.error(error);
    setStatus("Webcam failed");
    resultsPanel.innerHTML = `
      <div class="result-miss">
        Could not start webcam. Check camera permissions and HTTPS/localhost.
      </div>
    `;
  }
}

function stopWebcam() {
  isRunningWebcam = false;

  if (animationFrameId) {
    cancelAnimationFrame(animationFrameId);
    animationFrameId = null;
  }

  if (webcamStream) {
    webcamStream.getTracks().forEach((track) => track.stop());
    webcamStream = null;
  }

  video.srcObject = null;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("Ready");
}

uploadInput.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;

  await waitForImage(img);

  uploadedImageObj = img;
  uploadedPreview.src = url;
  uploadedPreview.style.display = "block";
  video.style.display = "none";
  runImageBtn.disabled = false;

  stopWebcam();
  setStatus("Image loaded");
  resultsPanel.innerHTML = `<div class="small">Image ready. Click "Detect In Uploaded Image".</div>`;
});

runImageBtn.addEventListener("click", () => {
  if (!uploadedImageObj || !imageEmbedder) return;

  const width = uploadedImageObj.naturalWidth;
  const height = uploadedImageObj.naturalHeight;

  const started = performance.now();
  prepareCanvasFromSource(uploadedImageObj, width, height);

  const proposals = generateProposals(
    width,
    height,
    grayCtx.getImageData(0, 0, width, height),
  );
  const threshold = Number(thresholdInput.value);

  const hits = [];

  for (const proposal of proposals) {
    const { canvas, sourceRect } = cropBoxToCanvas(frameCanvas, proposal, 0.15);
    const embedding = embedCanvas(canvas);
    if (!embedding) continue;

    const match = bestReferenceSimilarity(embedding);
    if (match.score >= threshold) {
      hits.push({
        x: sourceRect.x,
        y: sourceRect.y,
        w: sourceRect.w,
        h: sourceRect.h,
        score: match.score,
      });
    }
  }

  const finalDetections = nonMaxSuppression(hits, 0.3);

  overlay.width = width;
  overlay.height = height;
  drawDetections(finalDetections, width, height);
  renderResults(finalDetections, performance.now() - started, proposals.length);
  setStatus("Image detection complete");
});

startBtn.addEventListener("click", startWebcam);
stopBtn.addEventListener("click", stopWebcam);

initMediaPipe().catch((error) => {
  console.error(error);
  setStatus("Initialization failed");
  resultsPanel.innerHTML = `
    <div class="result-miss">
      Failed to initialize MediaPipe.
    </div>
    <div class="small">
      Open the browser console and verify that star.png is present and the model URLs are reachable.
    </div>
  `;
});
