import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'

export default function PredictionViewer({ eoCanvas, sarCanvas, maskCanvas, heatmapCanvas, stats, overlayOpacity = 0.7 }) {
  const overlayRef = useRef(null)

  // Draw the EO image with mask overlay
  useEffect(() => {
    if (!overlayRef.current || !eoCanvas || !maskCanvas) return
    const canvas = overlayRef.current
    const ctx = canvas.getContext('2d')
    canvas.width = eoCanvas.width
    canvas.height = eoCanvas.height

    // Draw EO base
    ctx.drawImage(eoCanvas, 0, 0)

    // Draw mask overlay
    ctx.globalAlpha = overlayOpacity
    ctx.drawImage(maskCanvas, 0, 0)
    ctx.globalAlpha = 1.0
  }, [eoCanvas, maskCanvas, overlayOpacity])

  if (!eoCanvas) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6 }}
    >
      {/* Three-panel view */}
      <div className="prediction-panel">
        <div className="prediction-image">
          <img src={eoCanvas.toDataURL()} alt="EO Input" />
          <div className="prediction-image-label">Pre-Event Optical (EO)</div>
        </div>

        <div className="prediction-image">
          <img src={sarCanvas.toDataURL()} alt="SAR Input" />
          <div className="prediction-image-label">Post-Event SAR</div>
        </div>

        <div className="prediction-image">
          <canvas ref={overlayRef} style={{ width: '100%', height: 'auto', display: 'block' }} />
          <div className="prediction-image-label">Detection Result (overlay)</div>
        </div>
      </div>

      {/* Heatmap */}
      {heatmapCanvas && (
        <div style={{ marginTop: '1.5rem' }}>
          <div className="prediction-image" style={{ maxWidth: '340px' }}>
            <img src={heatmapCanvas.toDataURL()} alt="Probability Heatmap" />
            <div className="prediction-image-label">Change Probability Heatmap</div>
          </div>
        </div>
      )}

      {/* Stats */}
      {stats && (
        <div className="glass-card" style={{ marginTop: '1.5rem', padding: '1.2rem' }}>
          <h4 style={{ marginBottom: '1rem', fontSize: '0.95rem' }}>Inference Statistics</h4>
          <div className="inference-stats">
            <div className="inference-stat">
              <div className="inference-stat-dot" />
              <span style={{ fontSize: '0.85rem' }}>
                <strong>{stats.changePercentage}%</strong> pixels changed
              </span>
            </div>
            <div className="inference-stat">
              <div className="inference-stat-dot" />
              <span style={{ fontSize: '0.85rem' }}>
                <strong>{stats.changedPixels.toLocaleString()}</strong> of {stats.totalPixels.toLocaleString()} pixels
              </span>
            </div>
            <div className="inference-stat">
              <div className="inference-stat-dot warning" />
              <span style={{ fontSize: '0.85rem' }}>
                Avg confidence: <strong>{stats.avgConfidence}</strong>
              </span>
            </div>
            <div className="inference-stat">
              <div className="inference-stat-dot" />
              <span style={{ fontSize: '0.85rem' }}>
                Inference: <strong>{stats.inferenceTimeMs}ms</strong> | Total: <strong>{stats.totalTimeMs}ms</strong>
              </span>
            </div>
          </div>
        </div>
      )}
    </motion.div>
  )
}
