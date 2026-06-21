import { useState, useEffect } from 'react'
import { Link, useLocation } from 'react-router-dom'
import { motion } from 'framer-motion'

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false)
  const location = useLocation()

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 40)
    window.addEventListener('scroll', onScroll)
    return () => window.removeEventListener('scroll', onScroll)
  }, [])

  const links = [
    { to: '/', label: 'Home' },
    { to: '/detect', label: 'Detect' },
    { to: '/#architecture', label: 'Architecture' },
    { to: '/#results', label: 'Results' },
  ]

  return (
    <motion.nav
      className={`navbar ${scrolled ? 'scrolled' : ''}`}
      initial={{ y: -80 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
    >
      <div className="navbar-content">
        <Link to="/" className="navbar-logo">
          <div className="navbar-logo-icon">🛰</div>
          <span>EO-SAR Detection</span>
        </Link>

        <ul className="navbar-links">
          {links.map((link) => (
            <li key={link.to}>
              <Link
                to={link.to}
                style={
                  location.pathname === link.to
                    ? { color: 'var(--color-primary)' }
                    : {}
                }
              >
                {link.label}
              </Link>
            </li>
          ))}
        </ul>

        <Link to="/detect" className="btn btn-primary" style={{ padding: '0.5rem 1.2rem', fontSize: '0.85rem' }}>
          Run Detection
        </Link>
      </div>
    </motion.nav>
  )
}
