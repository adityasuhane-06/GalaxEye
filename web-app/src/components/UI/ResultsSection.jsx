import { useRef, useEffect, useState } from 'react'
import { motion } from 'framer-motion'

function MetricRing({ value, max = 1, label, color = 'var(--color-primary)', size = 120 }) {
  const [animatedValue, setAnimatedValue] = useState(0)
  const radius = (size - 12) / 2
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference * (1 - animatedValue / max)

  return (
    <motion.div
      className="glass-card"
      style={{ textAlign: 'center', padding: '1.5rem' }}
      initial={{ opacity: 0, y: 30 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ duration: 0.6 }}
      onViewportEnter={() => {
        setTimeout(() => setAnimatedValue(value), 200)
      }}
    >
      <div className="metric-ring" style={{ width: size, height: size }}>
        <svg width={size} height={size}>
          {/* Background circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.05)"
            strokeWidth="6"
          />
          {/* Value circle */}
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="6"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            style={{ transition: 'stroke-dashoffset 1.5s cubic-bezier(0.4, 0, 0.2, 1)' }}
          />
        </svg>
        <div className="metric-ring-value" style={{ color }}>
          {animatedValue.toFixed(2)}
        </div>
      </div>
      <div style={{ fontWeight: 600, fontSize: '0.9rem', marginTop: '0.5rem' }}>{label}</div>
    </motion.div>
  )
}

const perSceneVal = [
  { scene: 'Scene 01', iou: 0.36, type: 'Cloud contamination' },
  { scene: 'Scene 02', iou: 0.79, type: 'Dense urban coastal' },
  { scene: 'Scene 03', iou: 0.79, type: 'Consistent performance' },
  { scene: 'Scene 04', iou: 0.10, type: 'Sparse, few buildings' },
  { scene: 'Scene 05', iou: 0.52, type: 'Moderate density' },
  { scene: 'Scene 06', iou: 0.86, type: 'Dense urban (best)' },
  { scene: 'Scene 07', iou: 0.47, type: 'Mixed terrain' },
  { scene: 'Scene 08', iou: 0.57, type: 'Moderate urban' },
]

const perSceneTest = [
  { scene: 'Scene 09', iou: 0.38, type: 'Good transfer' },
  { scene: 'Scene 10', iou: 0.04, type: 'Poor transfer' },
]

function IoUBar({ scene, iou, type, maxIoU = 1 }) {
  const [width, setWidth] = useState(0)
  const barColor = iou > 0.6 ? 'var(--color-accent)' : iou > 0.3 ? 'var(--color-primary)' : 'var(--color-danger)'

  return (
    <motion.div
      style={{ marginBottom: '0.8rem' }}
      initial={{ opacity: 0, x: -20 }}
      whileInView={{ opacity: 1, x: 0 }}
      viewport={{ once: true }}
      onViewportEnter={() => setTimeout(() => setWidth(iou / maxIoU * 100), 100)}
    >
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.3rem' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 500 }}>{scene}</span>
        <span className="mono" style={{ fontSize: '0.8rem', color: barColor }}>{iou.toFixed(2)}</span>
      </div>
      <div style={{
        background: 'var(--color-surface)',
        borderRadius: 'var(--radius-full)',
        height: '8px',
        overflow: 'hidden'
      }}>
        <div style={{
          height: '100%',
          width: `${width}%`,
          background: barColor,
          borderRadius: 'var(--radius-full)',
          transition: 'width 1s cubic-bezier(0.4, 0, 0.2, 1)',
        }} />
      </div>
      <span style={{ fontSize: '0.7rem', color: 'var(--color-text-muted)' }}>{type}</span>
    </motion.div>
  )
}

export default function ResultsSection() {
  return (
    <section className="section" id="results" style={{ background: 'var(--color-bg-secondary)' }}>
      <div className="container">
        <div className="section-header">
          <span className="badge">Results</span>
          <h2 className="heading-lg">
            Performance <span className="text-gradient">Metrics</span>
          </h2>
          <p>5.3x improvement over baseline, with detailed per-scene analysis.</p>
        </div>

        {/* Main metrics */}
        <div className="results-grid">
          <MetricRing value={0.6486} label="Validation IoU" color="var(--color-accent)" />
          <MetricRing value={0.3032} label="Test IoU" color="var(--color-primary)" />
          <MetricRing value={0.7869} label="F1 Score" color="var(--color-gradient-start)" />
          <MetricRing value={0.8647} label="Recall" color="var(--color-warning)" />
        </div>

        {/* Per-scene breakdown */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '2rem', marginTop: '3rem' }}>
          <motion.div
            className="glass-card"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h3 className="heading-md" style={{ marginBottom: '1.5rem', fontSize: '1.1rem' }}>
              Validation (Seen Events)
            </h3>
            {perSceneVal.map((s) => (
              <IoUBar key={s.scene} {...s} />
            ))}
          </motion.div>

          <motion.div
            className="glass-card"
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
          >
            <h3 className="heading-md" style={{ marginBottom: '1.5rem', fontSize: '1.1rem' }}>
              Test (Unseen Events)
            </h3>
            {perSceneTest.map((s) => (
              <IoUBar key={s.scene} {...s} />
            ))}
            <div style={{
              marginTop: '2rem',
              padding: '1rem',
              background: 'rgba(255, 56, 96, 0.08)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid rgba(255, 56, 96, 0.2)',
              fontSize: '0.8rem',
              color: 'var(--color-text-secondary)',
            }}>
              <strong style={{ color: 'var(--color-danger)' }}>Domain Shift:</strong> Val to Test IoU drop (0.65 to 0.30)
              is caused by cross-event generalization. The model struggles with unseen disaster types and building architectures.
            </div>
          </motion.div>
        </div>

        {/* Training stats */}
        <motion.div
          className="glass-card"
          style={{ marginTop: '2rem', padding: '1.5rem 2rem' }}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1.5rem' }}>
            {[
              { label: 'Epochs', value: '80' },
              { label: 'Training Time', value: '3.15 hrs' },
              { label: 'Best Epoch', value: '79' },
              { label: 'Threshold', value: '0.7' },
              { label: 'Hardware', value: '2× T4 GPU' },
              { label: 'Batch Size', value: '16' },
            ].map((item) => (
              <div key={item.label} style={{ textAlign: 'center' }}>
                <div className="mono" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--color-primary)' }}>
                  {item.value}
                </div>
                <div style={{ fontSize: '0.75rem', color: 'var(--color-text-muted)', marginTop: '0.25rem' }}>
                  {item.label}
                </div>
              </div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
