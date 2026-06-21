import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export default function SatelliteOrbit() {
  const satRef = useRef()
  const beamRef = useRef()
  const trailRef = useRef()

  useFrame((state) => {
    const t = state.clock.getElapsedTime()
    const radius = 3.2
    const speed = 0.3

    if (satRef.current) {
      satRef.current.position.x = Math.cos(t * speed) * radius
      satRef.current.position.z = Math.sin(t * speed) * radius
      satRef.current.position.y = Math.sin(t * speed * 0.7) * 0.8
      satRef.current.rotation.y = -t * speed + Math.PI / 2
    }

    if (beamRef.current) {
      beamRef.current.position.copy(satRef.current.position)
      const dir = new THREE.Vector3()
        .subVectors(new THREE.Vector3(0, 0, 0), satRef.current.position)
        .normalize()
      beamRef.current.lookAt(0, 0, 0)
      beamRef.current.material.opacity = 0.15 + Math.sin(t * 3) * 0.08
    }

    if (trailRef.current) {
      const positions = trailRef.current.geometry.attributes.position.array
      for (let i = 0; i < 30; i++) {
        const age = i / 30
        const tt = t * speed - age * 0.5
        positions[i * 3] = Math.cos(tt) * radius * (1 - age * 0.02)
        positions[i * 3 + 1] = Math.sin(tt * 0.7) * 0.8 * (1 - age * 0.1)
        positions[i * 3 + 2] = Math.sin(tt) * radius * (1 - age * 0.02)
      }
      trailRef.current.geometry.attributes.position.needsUpdate = true
    }
  })

  return (
    <group>
      {/* Satellite body */}
      <group ref={satRef}>
        {/* Main body */}
        <mesh>
          <boxGeometry args={[0.08, 0.06, 0.12]} />
          <meshStandardMaterial color="#c0c0c0" metalness={0.8} roughness={0.2} />
        </mesh>
        {/* Solar panel left */}
        <mesh position={[-0.18, 0, 0]}>
          <boxGeometry args={[0.22, 0.01, 0.08]} />
          <meshStandardMaterial color="#1a237e" metalness={0.5} roughness={0.3} emissive="#0d47a1" emissiveIntensity={0.3} />
        </mesh>
        {/* Solar panel right */}
        <mesh position={[0.18, 0, 0]}>
          <boxGeometry args={[0.22, 0.01, 0.08]} />
          <meshStandardMaterial color="#1a237e" metalness={0.5} roughness={0.3} emissive="#0d47a1" emissiveIntensity={0.3} />
        </mesh>
        {/* Antenna */}
        <mesh position={[0, 0.06, 0]}>
          <cylinderGeometry args={[0.005, 0.005, 0.08, 6]} />
          <meshStandardMaterial color="#ffffff" emissive="#00e5ff" emissiveIntensity={0.5} />
        </mesh>
        {/* Signal indicator */}
        <pointLight color="#00e5ff" intensity={0.5} distance={1} />
      </group>

      {/* Scan beam (cone from satellite to earth) */}
      <mesh ref={beamRef}>
        <coneGeometry args={[0.6, 2.8, 16, 1, true]} />
        <meshBasicMaterial color="#00e5ff" transparent opacity={0.12} side={THREE.DoubleSide} wireframe />
      </mesh>

      {/* Trail */}
      <line ref={trailRef}>
        <bufferGeometry>
          <bufferAttribute
            attach="attributes-position"
            count={30}
            array={new Float32Array(90)}
            itemSize={3}
          />
        </bufferGeometry>
        <lineBasicMaterial color="#00e5ff" transparent opacity={0.3} />
      </line>

      {/* Orbit ring */}
      <mesh rotation={[0, 0, 0]}>
        <torusGeometry args={[3.2, 0.003, 4, 128]} />
        <meshBasicMaterial color="#00e5ff" transparent opacity={0.1} />
      </mesh>
    </group>
  )
}
