import React from 'react'
import { motion } from 'framer-motion'

const stages = [
  {
    icon: '🏗',
    title: 'Stage 1: Building Extraction',
    subtitle: 'BuildingExtractor',
    input: 'Pre-Event EO (3ch)',
    output: 'Building Probability Map',
    details: 'ResNet34 U-Net encoder-decoder processes only optical imagery to identify building footprints.',
    color: 'var(--color-primary)',
  },
  {
    icon: '💥',
    title: 'Stage 2: Damage Classification',
    subtitle: 'DamageClassifier',
    input: 'EO (3ch) + SAR (3ch)',
    output: 'Damage Probability Map',
    details: 'Late-fusion U-Net with separate ResNet34 encoders for EO and SAR. Features fused at decoder level.',
    color: 'var(--color-accent)',
  },
  {
    icon: '🎯',
    title: 'Final Output',
    subtitle: 'Element-wise Product',
    input: 'Building Prob × Damage Prob',
    output: 'Change Probability Map',
    details: 'Hard constraint: change can only occur where buildings exist. Zero building probability forces zero change.',
    color: 'var(--color-warning)',
  },
]

const failedArchitectures = [
  { name: 'ResNetUNet (4ch)', iou: '<0.06', reason: 'Cannot learn modality-specific features' },
  { name: 'LateFusionUNet', iou: '<0.06', reason: 'Background overwhelms the loss signal' },
  { name: 'SharedSiameseUNet', iou: '<0.05', reason: 'Shared weights too restrictive for cross-modal' },
  { name: 'CrossModalGatedDiff', iou: '~0.057', reason: 'Still fights 85% background noise' },
]

const cardVariants = {
  hidden: { opacity: 0, y: 40 },
  visible: (i) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.2, duration: 0.6, ease: 'easeOut' },
  }),
}

export default function ArchitectureSection() {
  return (
    <section className="section" id="architecture">
      <div className="container">
        <div className="section-header">
          <span className="badge">Architecture</span>
          <h2 className="heading-lg">
            Building-Guided <span className="text-gradient">Two-Stage</span> Model
          </h2>
          <p>
            A single end-to-end nn.Module with two internal stages, trained jointly
            in one forward pass. One model file, one checkpoint.
          </p>
        </div>

        {/* Pipeline */}
        <div className="arch-pipeline">
          {stages.map((stage, i) => (
            <React.Fragment key={stage.title}>
              {i > 0 && (
                <div className="arch-arrow">→</div>
              )}
              <motion.div
                className="glass-card arch-card"
                custom={i}
                variants={cardVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true, margin: '-50px' }}
              >
                <div className="arch-card-icon">{stage.icon}</div>
                <h3 className="heading-md" style={{ fontSize: '1rem' }}>{stage.title}</h3>
                <p className="mono" style={{ fontSize: '0.75rem', color: stage.color, margin: '0.5rem 0' }}>
                  {stage.subtitle}
                </p>
                <div style={{ fontSize: '0.8rem', color: 'var(--color-text-secondary)', marginBottom: '0.5rem' }}>
                  <div><strong>In:</strong> {stage.input}</div>
                  <div><strong>Out:</strong> {stage.output}</div>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--color-text-muted)', lineHeight: 1.5 }}>
                  {stage.details}
                </p>
              </motion.div>
            </React.Fragment>
          ))}
        </div>

        {/* Formula */}
        <motion.div
          className="arch-formula"
          initial={{ opacity: 0, scale: 0.95 }}
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
        >
          change_prob = σ(building_logits) × σ(damage_logits)
        </motion.div>

        {/* Previous approaches */}
        <motion.div
          style={{ marginTop: '4rem' }}
          initial={{ opacity: 0 }}
          whileInView={{ opacity: 1 }}
          viewport={{ once: true }}
          transition={{ duration: 0.6 }}
        >
          <h3 className="heading-md" style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
            Why This Architecture? <span className="text-secondary" style={{ fontWeight: 400 }}>Previous Approaches Failed</span>
          </h3>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gap: '1rem' }}>
            {failedArchitectures.map((arch, i) => (
              <motion.div
                className="glass-card"
                key={arch.name}
                custom={i}
                variants={cardVariants}
                initial="hidden"
                whileInView="visible"
                viewport={{ once: true }}
                style={{ padding: '1.2rem' }}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{arch.name}</span>
                  <span className="mono" style={{ color: 'var(--color-danger)', fontSize: '0.8rem' }}>
                    IoU {arch.iou}
                  </span>
                </div>
                <p style={{ fontSize: '0.78rem', color: 'var(--color-text-secondary)' }}>
                  {arch.reason}
                </p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </section>
  )
}
