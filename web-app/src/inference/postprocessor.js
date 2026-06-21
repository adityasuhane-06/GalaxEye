/**
 * Postprocess model output: threshold, colorize, and generate stats.
 */

const DEFAULT_THRESHOLD = 0.7

/**
 * Apply threshold and generate a colored overlay mask.
 * @param {Float32Array} changeProbData - Raw output (1, 1, H, W)
 * @param {number} width
 * @param {number} height
 * @param {number} threshold
 * @returns {{canvas: HTMLCanvasElement, stats: Object}}
 */
export function postprocessPrediction(changeProbData, width, height, threshold = DEFAULT_THRESHOLD) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  const imageData = ctx.createImageData(width, height)

  let positivePixels = 0
  let totalConfidence = 0
  let maxConfidence = 0

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = y * width + x
      const prob = changeProbData[idx]
      const pixelIdx = idx * 4

      if (prob > threshold) {
        // Change detected - red overlay with opacity proportional to confidence
        const intensity = Math.min(1, (prob - threshold) / (1 - threshold) + 0.3)
        imageData.data[pixelIdx] = 255      // R
        imageData.data[pixelIdx + 1] = 50   // G
        imageData.data[pixelIdx + 2] = 50   // B
        imageData.data[pixelIdx + 3] = Math.floor(intensity * 200) // A
        positivePixels++
        totalConfidence += prob
        maxConfidence = Math.max(maxConfidence, prob)
      } else {
        // No change - transparent
        imageData.data[pixelIdx] = 0
        imageData.data[pixelIdx + 1] = 0
        imageData.data[pixelIdx + 2] = 0
        imageData.data[pixelIdx + 3] = 0
      }
    }
  }

  ctx.putImageData(imageData, 0, 0)

  const totalPixels = width * height
  const stats = {
    changedPixels: positivePixels,
    totalPixels,
    changePercentage: ((positivePixels / totalPixels) * 100).toFixed(2),
    avgConfidence: positivePixels > 0 ? (totalConfidence / positivePixels).toFixed(3) : 0,
    maxConfidence: maxConfidence.toFixed(3),
    threshold,
  }

  return { canvas, stats }
}

/**
 * Generate a heatmap canvas from raw probability values.
 * @param {Float32Array} probData
 * @param {number} width
 * @param {number} height
 * @returns {HTMLCanvasElement}
 */
export function generateHeatmap(probData, width, height) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  const ctx = canvas.getContext('2d')
  const imageData = ctx.createImageData(width, height)

  for (let i = 0; i < width * height; i++) {
    const prob = probData[i]
    const pixelIdx = i * 4

    // Blue (low) -> Cyan -> Green -> Yellow -> Red (high)
    let r, g, b
    if (prob < 0.25) {
      r = 0; g = Math.floor(prob * 4 * 255); b = 255
    } else if (prob < 0.5) {
      r = 0; g = 255; b = Math.floor((1 - (prob - 0.25) * 4) * 255)
    } else if (prob < 0.75) {
      r = Math.floor((prob - 0.5) * 4 * 255); g = 255; b = 0
    } else {
      r = 255; g = Math.floor((1 - (prob - 0.75) * 4) * 255); b = 0
    }

    imageData.data[pixelIdx] = r
    imageData.data[pixelIdx + 1] = g
    imageData.data[pixelIdx + 2] = b
    imageData.data[pixelIdx + 3] = Math.floor(Math.max(prob * 255, 20))
  }

  ctx.putImageData(imageData, 0, 0)
  return canvas
}
