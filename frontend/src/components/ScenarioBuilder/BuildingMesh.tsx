import { useMemo } from 'react'
import * as THREE from 'three'
import { BuildingGeoJSON } from '../../types/city'
import { ClimateData } from '../../types/climate'
import { getClimateColor } from './ClimateOverlay'

interface BuildingMeshProps {
  buildingsGeoJSON: BuildingGeoJSON
  climateData?: ClimateData[]
  overlayType?: 'temperature' | 'humidity' | 'precipitation'
}

export default function BuildingMesh({
  buildingsGeoJSON,
  climateData,
  overlayType = 'temperature',
}: BuildingMeshProps) {
  const latestClimate = useMemo(() => {
    if (!climateData || climateData.length === 0) return undefined
    return climateData[0] // Most recent data point
  }, [climateData])

  const baseColor = useMemo(() => {
    return getClimateColor(latestClimate, overlayType)
  }, [latestClimate, overlayType])

  const meshes = useMemo(() => {
    if (!buildingsGeoJSON?.features) return []

    return buildingsGeoJSON.features.map((feature, index) => {
      const geometry = feature.geometry
      const properties = feature.properties
      const height = properties.height || 10 // Default height if not provided

      if (geometry.type !== 'Polygon' && geometry.type !== 'MultiPolygon') {
        return null
      }

      const polygons = geometry.type === 'Polygon' 
        ? [geometry.coordinates] 
        : geometry.coordinates

      const buildingMeshes: JSX.Element[] = []

      polygons.forEach((polygon, polyIndex) => {
        const [exteriorRing, ...interiorRings] = polygon
        
        // Create shape from exterior ring
        const shape = new THREE.Shape()
        const firstPoint = exteriorRing[0] as number[]
        shape.moveTo(firstPoint[0], firstPoint[1])
        
        for (let i = 1; i < exteriorRing.length; i++) {
          const point = exteriorRing[i] as number[]
          shape.lineTo(point[0], point[1])
        }

        // Add holes for interior rings
        interiorRings.map(ring => {
          const hole = new THREE.Path()
          const firstHolePoint = ring[0] as number[]
          hole.moveTo(firstHolePoint[0], firstHolePoint[1])
          for (let i = 1; i < ring.length; i++) {
            const point = ring[i] as number[]
            hole.lineTo(point[0], point[1])
          }
          return hole
        })

        // Create extrude geometry
        const extrudeSettings = {
          depth: height,
          bevelEnabled: false,
        }
        const extrudeGeometry = new THREE.ExtrudeGeometry(shape, extrudeSettings)

        // Center the geometry
        extrudeGeometry.translate(0, 0, height / 2)

        buildingMeshes.push(
          <mesh
            key={`${index}-${polyIndex}`}
            geometry={extrudeGeometry}
            position={[0, 0, 0]}
          >
            <meshStandardMaterial
              color={baseColor}
              metalness={0.1}
              roughness={0.8}
            />
          </mesh>
        )
      })

      return buildingMeshes
    }).flat().filter(Boolean)
  }, [buildingsGeoJSON, baseColor])

  return <>{meshes}</>
}
