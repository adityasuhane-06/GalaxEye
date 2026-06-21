import { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import ImageUploader from './ImageUploader'
import PredictionViewer from './PredictionViewer'
import { runInference } from '../../inference/runInference'
import { isModelLoaded } from '../../inference/modelLoader'

export default function DetectPage() {
  const [eoFile, setEoFile] = useState(null)
  const [sarFile, setSarFile] = useState(null)
  const [eoPreview, setEoPreview] = useState(null)
  const [sarPreview, setSarPreview] = useState(null)
  const [threshold, setThreshold] = useState(0.7)
  const [overlayOpacity, setOverlayOpacity] = useState(0.7)
  const [isRunning, setIsRunning] = useState(false)
  const [progress, setProgress] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const handleEoSelect = useCallback((file) => {
    setEoFile(file)
    setEoPreview(URL.createObjectURL(file))
    setResult(null)
  }, [])

  const handleSarSelect = useCallback((file) => {
    setSarFile(file)
    setSarPreview(URL.createObjectURL(file))
    setResult(null)
  }, [])

  const handleRunDetection = async () => {
    if (!eoFile || !sarFile) return
    setIsRunning(true)
    setError(null)
    setResult(null)

    try {
      const res = await runInference(eoFile, sarFile, threshold, (p) => {
        setProgress(p)
      })
      setResult(res)
    } catch (err) {
      console.error('Inference error:', err)
      setError(err.message || 'Inference failed. Please check the console for details.')
    } finally {
      setIsRunning(false)
      setProgress(null)
    }
  }

  const canRun = eoFile && sarFile && !isRunning

  return (
    <div className="detect-page">
      <div className="container">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <span className="badge" style={{ marginBottom: '1rem', display: 'inline-flex' }}>
            Live Detection
          </span>
          <h1 className="heading-lg">
            Run <span className="text-gradient">Change Detection</span>
          </h1>
          <p className="text-secondary" style={{ marginTop: '0.5rem', maxWidth: '600px' }}>
            Upload a pre-event optical (EO) image and a post-event SAR image.
            The neural network runs entirely in your browser using ONNX Runtime Web.
          </p>
        </motion.div>

        {/* Upload area */}
        <div className="detect-grid">
          <ImageUploader
            label="Pre-Event Optical (EO)"
            icon="🌍"
            onFileSelect={handleEoSelect}
            preview={eoPreview}
          />
          <ImageUploader
            label="Post-Event SAR"
            icon="📡"
            onFileSelect={handleSarSelect}
            preview={sarPreview}
          />
        </div>

        {/* Controls */}
        <motion.div
          className="glass-card"
          style={{ marginTop: '1.5rem', padding: '1.2rem 1.5rem' }}
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem' }}>
            {/* Threshold slider */}
            <div className="slider-container" style={{ flex: 1, minWidth: '200px' }}>
              <span style={{ fontSize: '0.85rem', whiteSpace: 'nowrap' }}>Threshold:</span>
              <input
                type="range"
                min="0.1"
                max="0.95"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
              />
              <span className="mono" style={{ fontSize: '0.85rem', color: 'var(--color-primary)' }}>
                {threshold.toFixed(2)}
              </span>
            </div>

            {/* Run button */}
            <button
              className={`btn btn-primary ${canRun ? 'btn-glow' : ''}`}
              onClick={handleRunDetection}
              disabled={!canRun}
              style={{ opacity: canRun ? 1 : 0.4 }}
            >
              {isRunning ? (
                <>
                  <div className="spinner" style={{ width: 18, height: 18, borderWidth: 2 }} />
                  {progress?.step || 'Processing...'}
                </>
              ) : (
                '🔍 Run Detection'
              )}
            </button>
          </div>

          {/* Progress bar */}
          <AnimatePresence>
            {progress && (
              <motion.div
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
              >
                <div className="progress-bar">
                  <div
                    className="progress-bar-fill"
                    style={{ width: `${(progress.progress || 0) * 100}%` }}
                  />
                </div>
                <p style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.5rem' }}>
                  {progress.step}
                </p>
              </motion.div>
            )}
          </AnimatePresence>
        </motion.div>

        {/* Error */}
        <AnimatePresence>
          {error && (
            <motion.div
              className="glass-card"
              style={{
                marginTop: '1.5rem',
                padding: '1rem',
                borderColor: 'rgba(255, 56, 96, 0.3)',
                background: 'rgba(255, 56, 96, 0.05)',
              }}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
            >
              <p style={{ color: 'var(--color-danger)', fontSize: '0.9rem' }}>
                <strong>Error:</strong> {error}
              </p>
              <p style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginTop: '0.5rem' }}>
                Make sure the ONNX model file (<code>model.onnx</code>) is available at{' '}
                <code>/model.onnx</code> in the public folder.
              </p>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Results */}
        {result && (
          <div style={{ marginTop: '2rem' }}>
            <h2 className="heading-md" style={{ marginBottom: '0.5rem' }}>
              Detection <span className="text-gradient">Results</span>
            </h2>

            {/* Overlay opacity slider */}
            <div className="slider-container" style={{ maxWidth: '300px', marginBottom: '1rem' }}>
              <span style={{ fontSize: '0.85rem', whiteSpace: 'nowrap' }}>Overlay:</span>
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                value={overlayOpacity}
                onChange={(e) => setOverlayOpacity(parseFloat(e.target.value))}
              />
              <span className="mono" style={{ fontSize: '0.85rem' }}>
                {Math.round(overlayOpacity * 100)}%
              </span>
            </div>

            <PredictionViewer
              eoCanvas={result.eoCanvas}
              sarCanvas={result.sarCanvas}
              maskCanvas={result.maskCanvas}
              heatmapCanvas={result.heatmapCanvas}
              stats={result.stats}
              overlayOpacity={overlayOpacity}
            />

            {/* Download button */}
            <button
              className="btn btn-outline"
              style={{ marginTop: '1rem' }}
              onClick={() => {
                const link = document.createElement('a')
                link.download = 'prediction_mask.png'
                link.href = result.maskCanvas.toDataURL('image/png')
                link.click()
              }}
            >
              📥 Download Prediction Mask
            </button>
          </div>
        )}

        {/* Model info */}
        <motion.div
          className="glass-card"
          style={{ marginTop: '3rem', marginBottom: '3rem', padding: '1.5rem' }}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
        >
          <h3 style={{ fontSize: '0.95rem', marginBottom: '0.75rem' }}>How It Works</h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: '1rem' }}>
            {[
              { step: '1', title: 'Upload', desc: 'EO (optical) + SAR (radar) images' },
              { step: '2', title: 'Preprocess', desc: 'Resize to 512x512, normalize with dataset stats' },
              { step: '3', title: 'Inference', desc: 'ONNX Runtime runs 72M parameter model in browser' },
              { step: '4', title: 'Visualize', desc: 'Threshold and display change probability map' },
            ].map((item) => (
              <div key={item.step} style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                <div style={{
                  width: 28, height: 28, borderRadius: '50%', flexShrink: 0,
                  background: 'linear-gradient(135deg, var(--color-primary), var(--color-accent))',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.75rem', fontWeight: 700, color: 'var(--color-bg-primary)',
                }}>
                  {item.step}
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{item.title}</div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>{item.desc}</div>
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  )
}
