import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function EarthGlobe() {
  const meshRef = useRef()
  const atmosphereRef = useRef()
  const scanRingRef = useRef()

  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    if (meshRef.current) {
      meshRef.current.rotation.y = t * 0.08
    }
    if (atmosphereRef.current) {
      atmosphereRef.current.rotation.y = t * 0.05
      atmosphereRef.current.rotation.x = Math.sin(t * 0.3) * 0.05
    }
    if (scanRingRef.current) {
      scanRingRef.current.rotation.x = Math.PI / 2
      scanRingRef.current.rotation.z = t * 0.5
    }
  })

  const earthGradient = useMemo(() => {
    const canvas = document.createElement('canvas')
    canvas.width = 512
    canvas.height = 256
    const ctx = canvas.getContext('2d')

    // Deep ocean blue base
    ctx.fillStyle = '#0a1628'
    ctx.fillRect(0, 0, 512, 256)

    // Land masses (simplified procedural)
    const grad = ctx.createLinearGradient(0, 0, 512, 256)
    grad.addColorStop(0, '#0d2137')
    grad.addColorStop(0.3, '#132e4a')
    grad.addColorStop(0.5, '#0a1e35')
    grad.addColorStop(0.7, '#163a5c')
    grad.addColorStop(1, '#0d2137')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 512, 256)

    // City lights (random dots)
    for (let i = 0; i < 300; i++) {
      const x = Math.random() * 512
      const y = 40 + Math.random() * 176
      const size = Math.random() * 2
      ctx.fillStyle = `rgba(0, 229, 255, ${0.3 + Math.random() * 0.5})`
      ctx.fillRect(x, y, size, size)
    }

    // Grid lines
    ctx.strokeStyle = 'rgba(0, 229, 255, 0.08)'
    ctx.lineWidth = 0.5
    for (let i = 0; i < 12; i++) {
      ctx.beginPath()
      ctx.moveTo(0, (i / 12) * 256)
      ctx.lineTo(512, (i / 12) * 256)
      ctx.stroke()
    }
    for (let i = 0; i < 24; i++) {
      ctx.beginPath()
      ctx.moveTo((i / 24) * 512, 0)
      ctx.lineTo((i / 24) * 512, 256)
      ctx.stroke()
    }

    const texture = new THREE.CanvasTexture(canvas)
    texture.wrapS = THREE.RepeatWrapping
    texture.wrapT = THREE.RepeatWrapping
    return texture
  }, [])

  return (
    <group>
      {/* Earth sphere */}
      <mesh ref={meshRef}>
        <sphereGeometry args={[2, 64, 64]} />
        <meshStandardMaterial
          map={earthGradient}
          roughness={0.8}
          metalness={0.2}
          emissive="#001830"
          emissiveIntensity={0.3}
        />
      </mesh>

      {/* Atmosphere glow (Fresnel-like) */}
      <mesh ref={atmosphereRef} scale={[1.08, 1.08, 1.08]}>
        <sphereGeometry args={[2, 48, 48]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.06}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Outer atmosphere */}
      <mesh scale={[1.15, 1.15, 1.15]}>
        <sphereGeometry args={[2, 32, 32]} />
        <meshBasicMaterial
          color="#00e5ff"
          transparent
          opacity={0.02}
          side={THREE.BackSide}
        />
      </mesh>

      {/* Scan ring */}
      <mesh ref={scanRingRef}>
        <torusGeometry args={[2.6, 0.015, 8, 128]} />
        <meshBasicMaterial color="#00e5ff" transparent opacity={0.4} />
      </mesh>

      {/* Inner scan ring */}
      <mesh rotation={[Math.PI / 3, 0, 0]}>
        <torusGeometry args={[2.4, 0.01, 8, 128]} />
        <meshBasicMaterial color="#39ff14" transparent opacity={0.2} />
      </mesh>
    </group>
  )
}
