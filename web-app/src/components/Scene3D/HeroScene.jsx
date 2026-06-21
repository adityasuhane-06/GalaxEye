import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { Stars, OrbitControls } from '@react-three/drei'
import EarthGlobe from './EarthGlobe'
import SatelliteOrbit from './SatelliteOrbit'

export default function HeroScene() {
  return (
    <div className="hero-canvas">
      <Canvas
        camera={{ position: [0, 1.5, 6], fov: 45 }}
        dpr={[1, 1.5]}
        gl={{ antialias: true, alpha: true }}
        style={{ background: 'transparent' }}
      >
        <Suspense fallback={null}>
          {/* Lighting */}
          <ambientLight intensity={0.15} />
          <directionalLight position={[5, 3, 5]} intensity={0.8} color="#e8eaf6" />
          <pointLight position={[-5, -2, -5]} intensity={0.3} color="#00e5ff" />

          {/* Stars background */}
          <Stars
            radius={100}
            depth={60}
            count={3000}
            factor={4}
            saturation={0}
            fade
            speed={0.5}
          />

          {/* Earth */}
          <EarthGlobe />

          {/* Satellite */}
          <SatelliteOrbit />

          {/* Controls */}
          <OrbitControls
            enableZoom={false}
            enablePan={false}
            autoRotate={false}
            maxPolarAngle={Math.PI / 1.5}
            minPolarAngle={Math.PI / 3}
          />
        </Suspense>
      </Canvas>
    </div>
  )
}
