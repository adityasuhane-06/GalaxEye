import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/UI/Navbar'
import HeroSection from './components/UI/HeroSection'
import ArchitectureSection from './components/UI/ArchitectureSection'
import ResultsSection from './components/UI/ResultsSection'
import Footer from './components/UI/Footer'
import DetectPage from './components/Detect/DetectPage'

function HomePage() {
  return (
    <>
      <HeroSection />
      <ArchitectureSection />
      <ResultsSection />
    </>
  )
}

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <AnimatePresence mode="wait">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/detect" element={<DetectPage />} />
        </Routes>
      </AnimatePresence>
      <Footer />
    </BrowserRouter>
  )
}
