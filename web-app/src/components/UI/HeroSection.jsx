import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import HeroScene from '../Scene3D/HeroScene'

const stats = [
  { value: '72M', label: 'Parameters' },
  { value: '0.65', label: 'Val IoU' },
  { value: '5.3×', label: 'Improvement' },
]

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: { staggerChildren: 0.15, delayChildren: 0.3 },
  },
}

const itemVariants = {
  hidden: { opacity: 0, y: 30 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.6, ease: 'easeOut' } },
}

export default function HeroSection() {
  return (
    <section className="hero" id="hero">
      <HeroScene />

      <motion.div
        className="hero-content container"
        variants={containerVariants}
        initial="hidden"
        animate="visible"
      >
        <motion.div variants={itemVariants}>
          <span className="badge">🛰 Satellite AI Research</span>
        </motion.div>

        <motion.h1 className="heading-xl" variants={itemVariants}>
          Building-Guided{' '}
          <span className="text-gradient">EO-SAR</span>
          <br />
          Change Detection System
        </motion.h1>

        <motion.p className="hero-subtitle" variants={itemVariants}>
          A two-stage deep learning model that combines optical (EO) and synthetic
          aperture radar (SAR) imagery to detect building damage at pixel level.
          Powered by in-browser neural network inference.
        </motion.p>

        <motion.div className="hero-stats" variants={itemVariants}>
          {stats.map((stat) => (
            <div key={stat.label}>
              <div className="stat-value text-gradient">{stat.value}</div>
              <div className="stat-label">{stat.label}</div>
            </div>
          ))}
        </motion.div>

        <motion.div className="hero-actions" variants={itemVariants}>
          <Link to="/detect" className="btn btn-primary btn-glow">
            🔍 Run Detection
          </Link>
          <a
            href="https://github.com/adityasuhane-06/GalaxEye"
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-outline"
          >
            ⭐ GitHub
          </a>
        </motion.div>
      </motion.div>
    </section>
  )
}
