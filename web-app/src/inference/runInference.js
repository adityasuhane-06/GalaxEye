import * as ort from 'onnxruntime-web'
import { loadModel } from './modelLoader'
import { preprocessEO, preprocessSAR, TARGET_SIZE } from './preprocessor'
import { postprocessPrediction, generateHeatmap } from './postprocessor'

/**
 * Run full inference pipeline: preprocess -> model -> postprocess.
 * @param {File} eoFile - EO image file
 * @param {File} sarFile - SAR image file
 * @param {number} threshold - Detection threshold (default 0.7)
 * @param {function} onProgress - Progress callback
 * @returns {Promise<{maskCanvas, heatmapCanvas, eoCanvas, sarCanvas, stats, inferenceTimeMs}>}
 */
export async function runInference(eoFile, sarFile, threshold = 0.7, onProgress) {
  const startTime = performance.now()

  // Step 1: Load model
  if (onProgress) onProgress({ step: 'Loading model...', progress: 0.1 })
  const session = await loadModel((p) => {
    if (onProgress) onProgress({ step: 'Loading model...', progress: p * 0.3 })
  })

  // Step 2: Preprocess images
  if (onProgress) onProgress({ step: 'Preprocessing images...', progress: 0.35 })
  const [eoResult, sarResult] = await Promise.all([
    preprocessEO(eoFile),
    preprocessSAR(sarFile),
  ])

  // Step 3: Create ONNX tensors
  if (onProgress) onProgress({ step: 'Preparing tensors...', progress: 0.5 })
  const eoTensor = new ort.Tensor('float32', eoResult.tensor, [1, 3, TARGET_SIZE, TARGET_SIZE])
  const sarTensor = new ort.Tensor('float32', sarResult.tensor, [1, 3, TARGET_SIZE, TARGET_SIZE])

  // Step 4: Run inference
  if (onProgress) onProgress({ step: 'Running neural network...', progress: 0.6 })
  const inferenceStart = performance.now()
  const feeds = { eo: eoTensor, sar: sarTensor }
  const results = await session.run(feeds)
  const inferenceEnd = performance.now()

  // Step 5: Postprocess
  if (onProgress) onProgress({ step: 'Generating predictions...', progress: 0.85 })
  const changeProbData = results.change_prob.data
  const { canvas: maskCanvas, stats } = postprocessPrediction(
    changeProbData, TARGET_SIZE, TARGET_SIZE, threshold
  )
  const heatmapCanvas = generateHeatmap(changeProbData, TARGET_SIZE, TARGET_SIZE)

  const totalTime = performance.now() - startTime
  const inferenceTimeMs = inferenceEnd - inferenceStart

  if (onProgress) onProgress({ step: 'Complete!', progress: 1.0 })

  return {
    maskCanvas,
    heatmapCanvas,
    eoCanvas: eoResult.canvas,
    sarCanvas: sarResult.canvas,
    stats: {
      ...stats,
      inferenceTimeMs: Math.round(inferenceTimeMs),
      totalTimeMs: Math.round(totalTime),
    },
  }
}
