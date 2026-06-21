import * as ort from 'onnxruntime-web'

let session = null
let loading = false

/**
 * Load the ONNX model. Caches the session after first load.
 * @param {function} onProgress - Callback for loading progress (0-1)
 * @returns {Promise<ort.InferenceSession>}
 */
export async function loadModel(onProgress) {
  if (session) return session
  if (loading) {
    // Wait for existing load
    while (loading) await new Promise((r) => setTimeout(r, 100))
    return session
  }

  loading = true
  try {
    if (onProgress) onProgress(0.1)

    // Configure WASM execution provider
    ort.env.wasm.numThreads = navigator.hardwareConcurrency || 4
    ort.env.wasm.simd = true

    if (onProgress) onProgress(0.3)

    session = await ort.InferenceSession.create('/model.onnx', {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    })

    if (onProgress) onProgress(1.0)
    console.log('Model loaded successfully')
    console.log('  Input names:', session.inputNames)
    console.log('  Output names:', session.outputNames)

    return session
  } catch (err) {
    console.error('Failed to load model:', err)
    throw err
  } finally {
    loading = false
  }
}

/**
 * Check if the model is loaded.
 */
export function isModelLoaded() {
  return session !== null
}

/**
 * Get the loaded session.
 */
export function getSession() {
  return session
}
