export default function Footer() {
  const techStack = [
    'React', 'Three.js', 'React Three Fiber', 'Drei',
    'Framer Motion', 'ONNX Runtime Web', 'Vite', 'ResNet34', 'U-Net',
  ]

  return (
    <footer className="footer">
      <div className="container">
        <ul className="footer-links">
          <li><a href="https://github.com/adityasuhane-06/GalaxEye" target="_blank" rel="noopener noreferrer">GitHub</a></li>
          <li><a href="https://drive.google.com/file/d/1dj8ZG_VhGZfQlfY-IeU-NPFmA1Pepivr/view?usp=sharing" target="_blank" rel="noopener noreferrer">Model Weights</a></li>
        </ul>

        <div className="tech-badges">
          {techStack.map((tech) => (
            <span className="tech-badge" key={tech}>{tech}</span>
          ))}
        </div>

        <p style={{ marginTop: '1.5rem' }}>
          Building-Guided EO-SAR Change Detection System
          <br />
          Built by <strong style={{ color: 'var(--color-text)' }}>Aditya Suhane</strong> &middot; 2026
        </p>
      </div>
    </footer>
  )
}
