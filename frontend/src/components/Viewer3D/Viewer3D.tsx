import { Suspense, useState } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { useCities, useCityBuildings, useCityClimate } from '../../hooks/useCities'
import CitySelector from '../Dashboard/CitySelector'
import BuildingMesh from './BuildingMesh'
import { FadeIn, SplitText, HoverCard } from '../reactbits'
import { ClimateData } from '../../types/climate'
import Iridescence from '../Iridescence'

export default function Viewer3D() {
  const [selectedCityId, setSelectedCityId] = useState<number | null>(null)
  const [overlayType, setOverlayType] = useState<'temperature' | 'humidity' | 'precipitation'>('temperature')
  const { data: cities } = useCities()
  const { data: buildingsGeoJSON } = useCityBuildings(selectedCityId || 0)
  const { data: climateData } = useCityClimate(selectedCityId || 0)

  const latestClimate = climateData?.[0]

  return (
    <div className="relative h-screen">
      <div className="fixed inset-0 z-0">
        <Iridescence />
      </div>
      <div className="relative z-10 px-4 py-6 h-screen flex flex-col">
      <FadeIn duration={800} delay={0}>
        <SplitText
          splitBy="word"
          stagger={50}
          direction="up"
          trigger="onMount"
          duration={600}
          className="text-3xl font-bold mb-4"
        >
          3D City Viewer
        </SplitText>
      </FadeIn>
      
      <FadeIn duration={800} delay={200}>
        <div className="mb-4 space-y-4">
          <CitySelector
            cities={cities || []}
            selectedCityId={selectedCityId}
            onSelectCity={setSelectedCityId}
          />

          {selectedCityId && (
            <div className="flex gap-4 items-center">
              <label className="text-sm font-medium text-gray-300">Overlay:</label>
              <select
                value={overlayType}
                onChange={(e) => setOverlayType(e.target.value as any)}
                className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
              >
                <option value="temperature">Temperature</option>
                <option value="humidity">Humidity</option>
                <option value="precipitation">Precipitation</option>
              </select>

              {latestClimate && (
                <HoverCard hover shadow="sm" padding="sm" border className="inline-block">
                  <div className="text-sm">
                    <span className="text-gray-400">
                      {overlayType === 'temperature' && `Temp: ${latestClimate.temperature.toFixed(1)}°C`}
                      {overlayType === 'humidity' && `Humidity: ${latestClimate.humidity?.toFixed(1) || 'N/A'}%`}
                      {overlayType === 'precipitation' && `Precip: ${latestClimate.precipitation?.toFixed(1) || 'N/A'}mm`}
                    </span>
                  </div>
                </HoverCard>
              )}
            </div>
          )}
        </div>
      </FadeIn>

      <div className="flex-1 bg-gray-900 rounded-lg overflow-hidden border border-gray-700">
        {selectedCityId && buildingsGeoJSON ? (
          <Canvas>
            <Suspense fallback={null}>
              <PerspectiveCamera makeDefault position={[0, 100, 200]} />
              <ambientLight intensity={0.5} />
              <directionalLight position={[10, 10, 5]} intensity={1} />
              <OrbitControls enablePan={true} enableZoom={true} enableRotate={true} />
              <BuildingMesh
                buildingsGeoJSON={buildingsGeoJSON}
                climateData={climateData as ClimateData[]}
                overlayType={overlayType}
              />
            </Suspense>
          </Canvas>
        ) : (
          <div className="flex items-center justify-center h-full text-gray-400">
            {selectedCityId ? 'Loading buildings...' : 'Select a city to view 3D buildings'}
          </div>
        )}
      </div>
    </div>
    </div>
  )
}

