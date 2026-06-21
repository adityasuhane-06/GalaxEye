/**
 * Image preprocessing for the Building-Guided Change Detector.
 * Matches the exact pipeline from 04_dataset.py.
 */

// Dataset-specific normalization stats (from EDA)
const EO_MEAN = [0.3217, 0.3462, 0.2881]
const EO_STD = [0.2406, 0.2160, 0.2056]
const SAR_MEAN = [0.2053, 0.2053, 0.2053]
const SAR_STD = [0.1626, 0.1626, 0.1626]

const TARGET_SIZE = 512

/**
 * Load an image file into a canvas and extract pixel data.
 * @param {File} file - Image file (PNG, JPG, or TIFF)
 * @returns {Promise<{imageData: ImageData, width: number, height: number}>}
 */
export async function loadImageToCanvas(file) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => {
      const canvas = document.createElement('canvas')
      canvas.width = TARGET_SIZE
      canvas.height = TARGET_SIZE
      const ctx = canvas.getContext('2d')
      ctx.drawImage(img, 0, 0, TARGET_SIZE, TARGET_SIZE)
      const imageData = ctx.getImageData(0, 0, TARGET_SIZE, TARGET_SIZE)
      resolve({ imageData, width: TARGET_SIZE, height: TARGET_SIZE, canvas })
    }
    img.onerror = reject
    img.src = URL.createObjectURL(file)
  })
}

/**
 * Convert ImageData to a normalized Float32Array in NCHW format.
 * @param {ImageData} imageData - Raw pixel data (RGBA)
 * @param {number[]} mean - Per-channel mean
 * @param {number[]} std - Per-channel std
 * @param {boolean} isSAR - If true, use only the first channel and replicate to 3
 * @returns {Float32Array} - (1, 3, H, W) tensor
 */
export function imageDataToTensor(imageData, mean, std, isSAR = false) {
  const { data, width, height } = imageData
  const tensor = new Float32Array(1 * 3 * height * width)

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const pixelIdx = (y * width + x) * 4
      const r = data[pixelIdx] / 255.0
      const g = data[pixelIdx + 1] / 255.0
      const b = data[pixelIdx + 2] / 255.0

      if (isSAR) {
        // Use grayscale average, replicate to 3 channels
        const gray = (r + g + b) / 3.0
        for (let c = 0; c < 3; c++) {
          const normalized = (gray - mean[c]) / std[c]
          tensor[c * height * width + y * width + x] = normalized
        }
      } else {
        // RGB channels
        const channels = [r, g, b]
        for (let c = 0; c < 3; c++) {
          const normalized = (channels[c] - mean[c]) / std[c]
          tensor[c * height * width + y * width + x] = normalized
        }
      }
    }
  }

  return tensor
}

/**
 * Preprocess an EO image file for inference.
 * @param {File} file
 * @returns {Promise<{tensor: Float32Array, canvas: HTMLCanvasElement}>}
 */
export async function preprocessEO(file) {
  const { imageData, canvas } = await loadImageToCanvas(file)
  const tensor = imageDataToTensor(imageData, EO_MEAN, EO_STD, false)
  return { tensor, canvas }
}

/**
 * Preprocess a SAR image file for inference.
 * @param {File} file
 * @returns {Promise<{tensor: Float32Array, canvas: HTMLCanvasElement}>}
 */
export async function preprocessSAR(file) {
  const { imageData, canvas } = await loadImageToCanvas(file)
  const tensor = imageDataToTensor(imageData, SAR_MEAN, SAR_STD, true)
  return { tensor, canvas }
}

export { TARGET_SIZE }
