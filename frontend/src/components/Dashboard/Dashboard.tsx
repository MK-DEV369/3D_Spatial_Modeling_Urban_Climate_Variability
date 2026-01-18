
import { useState, useEffect, useRef } from 'react'
import { useCities, useCityClimate, useCityTraffic } from '../../hooks/useCities'
import { useOSMByBBox } from '../../hooks/useOSM'
import { MapContainer, TileLayer, GeoJSON, useMap, Rectangle, useMapEvents } from 'react-leaflet'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import OSMCRUD from './OSMCRUD'

// Component to handle map bounds changes and fetch OSM data
function MapBoundsHandler({ 
  onBoundsChange 
}: { 
  onBoundsChange: (bbox: { minLon: number; minLat: number; maxLon: number; maxLat: number }) => void 
}) {
  const map = useMap()
  const timeoutRef = useRef<number>()

  useEffect(() => {
    const updateBounds = () => {
      const bounds = map.getBounds()
      const bbox = {
        minLon: bounds.getWest(),
        minLat: bounds.getSouth(),
        maxLon: bounds.getEast(),
        maxLat: bounds.getNorth(),
      }
      onBoundsChange(bbox)
    }

    // Initial bounds
    updateBounds()

    // Update bounds on move/zoom (debounced)
    const handleMoveEnd = () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(updateBounds, 500)
    }

    map.on('moveend', handleMoveEnd)
    map.on('zoomend', handleMoveEnd)

    return () => {
      map.off('moveend', handleMoveEnd)
      map.off('zoomend', handleMoveEnd)
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [map, onBoundsChange])

  return null
}

// Component for area selection (rectangle drawing)
function AreaSelector({ 
  isSelecting, 
  onSelectionComplete 
}: { 
  isSelecting: boolean
  onSelectionComplete: (bbox: { minLon: number; minLat: number; maxLon: number; maxLat: number }) => void 
}) {
  const [startPoint, setStartPoint] = useState<[number, number] | null>(null)
  const [endPoint, setEndPoint] = useState<[number, number] | null>(null)
  const map = useMap()

  useMapEvents({
    click(e) {
      if (!isSelecting) return
      
      if (!startPoint) {
        setStartPoint([e.latlng.lat, e.latlng.lng])
      } else {
        setEndPoint([e.latlng.lat, e.latlng.lng])
        const minLat = Math.min(startPoint[0], e.latlng.lat)
        const maxLat = Math.max(startPoint[0], e.latlng.lat)
        const minLon = Math.min(startPoint[1], e.latlng.lng)
        const maxLon = Math.max(startPoint[1], e.latlng.lng)
        
        onSelectionComplete({ minLon, minLat, maxLon, maxLat })
        setStartPoint(null)
        setEndPoint(null)
      }
    },
    mousemove(e) {
      if (isSelecting && startPoint && !endPoint) {
        setEndPoint([e.latlng.lat, e.latlng.lng])
      }
    }
  })

  if (!isSelecting || !startPoint) return null

  const bounds = endPoint 
    ? [[Math.min(startPoint[0], endPoint[0]), Math.min(startPoint[1], endPoint[1])], 
       [Math.max(startPoint[0], endPoint[0]), Math.max(startPoint[1], endPoint[1])]]
    : null

  return bounds ? (
    <Rectangle
      bounds={bounds as L.LatLngBoundsExpression}
      pathOptions={{
        color: '#ff0000',
        weight: 2,
        fillColor: '#ff0000',
        fillOpacity: 0.2
      }}
    />
  ) : null
}

export default function Dashboard() {
  const [selectedCityId, setSelectedCityId] = useState<number | null>(null)
  const [activeTab, setActiveTab] = useState<'weather' | 'traffic' | 'urban' | 'water' | 'buildings' | 'roads' | 'green'>('weather')
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [mapBounds, setMapBounds] = useState<{ minLon: number; minLat: number; maxLon: number; maxLat: number } | null>(null)
  const [scenario, setScenario] = useState<string>('baseline')
  const [showBuildings, setShowBuildings] = useState(true)
  const [showRoads, setShowRoads] = useState(false)
  const [showWater, setShowWater] = useState(false)
  const [showGreen, setShowGreen] = useState(false)
  const [highlightedFeature, setHighlightedFeature] = useState<any>(null)
  const [highlightedOsmId, setHighlightedOsmId] = useState<number | null>(null)
  const [selectedArea, setSelectedArea] = useState<{ minLon: number; minLat: number; maxLon: number; maxLat: number } | null>(null)
  const [isSelectingArea, setIsSelectingArea] = useState(false)
  const [filterByArea, setFilterByArea] = useState(false)

  const { data: cities } = useCities()
  const { data: climateData } = useCityClimate(selectedCityId || 0)
  const { data: trafficData } = useCityTraffic(selectedCityId || 0)

  // Determine which bbox to use (selected area or map bounds)
  const activeBbox = filterByArea && selectedArea ? selectedArea : mapBounds

  // Fetch OSM data from backend based on map bounds or selected area
  const { data: buildingsData, isLoading: buildingsLoading } = useOSMByBBox(
    'buildings',
    activeBbox,
    { scenario, active: true, enabled: showBuildings && !!activeBbox }
  )

  const { data: roadsData, isLoading: roadsLoading } = useOSMByBBox(
    'roads',
    activeBbox,
    { scenario, active: true, enabled: showRoads && !!activeBbox }
  )

  const { data: waterData, isLoading: waterLoading } = useOSMByBBox(
    'water',
    activeBbox,
    { scenario, active: true, enabled: showWater && !!activeBbox }
  )

  const { data: greenData, isLoading: greenLoading } = useOSMByBBox(
    'green',
    activeBbox,
    { scenario, active: true, enabled: showGreen && !!activeBbox }
  )

  // Handle feature selection from OSMCRUD
  const handleFeatureSelect = (feature: any) => {
    setHighlightedFeature(feature)
    const osmId = feature.properties?.osm_id || feature.properties?.id
    if (osmId) {
      setHighlightedOsmId(osmId)
    }
  }

  const handleFeatureHighlight = (osmId: number | null) => {
    setHighlightedOsmId(osmId)
    if (!osmId) {
      setHighlightedFeature(null)
    }
  }

  const handleAreaSelection = (bbox: { minLon: number; minLat: number; maxLon: number; maxLat: number }) => {
    setSelectedArea(bbox)
    setIsSelectingArea(false)
    setFilterByArea(true)
  }

  const latestClimate = climateData?.[0]
  const latestTraffic = trafficData?.[0]
  const selectedCity = cities?.find(city => city.id === selectedCityId)

  // Set initial map bounds when city is selected
  useEffect(() => {
    if (selectedCity) {
      // Set initial bounds around the city (approximate 10km radius)
      const lat = selectedCity.latitude
      const lon = selectedCity.longitude
      const offset = 0.05 // ~5km
      setMapBounds({
        minLon: lon - offset,
        minLat: lat - offset,
        maxLon: lon + offset,
        maxLat: lat + offset,
      })
    }
  }, [selectedCity])


  return (
    <div className="fixed inset-0 flex bg-gray-900">
      {/* Full-screen OSM Buildings & Roads Visualization */}
      <div className="flex-1 relative">
        {selectedCity ? (
          <div className="w-full h-full bg-gray-900">
            <MapContainer
              center={[selectedCity.latitude, selectedCity.longitude]}
              zoom={13}
              style={{ height: '100%', width: '100%', zIndex: 0 }}
              key={`map-${selectedCity.id}`} // Force remount when city changes
            >
              <TileLayer
                url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
                attribution="&copy; OpenStreetMap contributors"
              />
              <MapBoundsHandler onBoundsChange={setMapBounds} />
              <AreaSelector 
                isSelecting={isSelectingArea} 
                onSelectionComplete={handleAreaSelection}
              />
              
              {/* Buildings Layer */}
              {showBuildings && buildingsData && (
                <GeoJSON 
                  key={`buildings-${highlightedOsmId || 'all'}`}
                  data={buildingsData} 
                  style={(feature) => {
                    const osmId = feature?.properties?.osm_id
                    const isHighlighted = osmId === highlightedOsmId
                    return {
                      color: isHighlighted ? '#ff0000' : '#ff9800',
                      weight: isHighlighted ? 3 : 1,
                      fillOpacity: isHighlighted ? 0.8 : 0.4,
                      fillColor: isHighlighted ? '#ff0000' : '#ff9800'
                    }
                  }}
                />
              )}
              
              {/* Roads Layer */}
              {showRoads && roadsData && (
                <GeoJSON 
                  key={`roads-${highlightedOsmId || 'all'}`}
                  data={roadsData} 
                  style={(feature) => {
                    const osmId = feature?.properties?.osm_id
                    const isHighlighted = osmId === highlightedOsmId
                    return {
                      color: isHighlighted ? '#ff0000' : '#2196f3',
                      weight: isHighlighted ? 4 : 2,
                      opacity: isHighlighted ? 1 : 0.8
                    }
                  }}
                />
              )}
              
              {/* Water Layer */}
              {showWater && waterData && (
                <GeoJSON 
                  key={`water-${highlightedOsmId || 'all'}`}
                  data={waterData} 
                  style={(feature) => {
                    const osmId = feature?.properties?.osm_id
                    const isHighlighted = osmId === highlightedOsmId
                    return {
                      color: isHighlighted ? '#ff0000' : '#03a9f4',
                      weight: isHighlighted ? 3 : 1,
                      fillOpacity: isHighlighted ? 0.8 : 0.5,
                      fillColor: isHighlighted ? '#ff0000' : '#03a9f4'
                    }
                  }}
                />
              )}
              
              {/* Green Spaces Layer */}
              {showGreen && greenData && (
                <GeoJSON 
                  key={`green-${highlightedOsmId || 'all'}`}
                  data={greenData} 
                  style={(feature) => {
                    const osmId = feature?.properties?.osm_id
                    const isHighlighted = osmId === highlightedOsmId
                    return {
                      color: isHighlighted ? '#ff0000' : '#4caf50',
                      weight: isHighlighted ? 3 : 1,
                      fillOpacity: isHighlighted ? 0.8 : 0.4,
                      fillColor: isHighlighted ? '#ff0000' : '#4caf50'
                    }
                  }}
                />
              )}
              
              {/* Selected Area Rectangle */}
              {selectedArea && filterByArea && (
                <Rectangle
                  bounds={[[selectedArea.minLat, selectedArea.minLon], [selectedArea.maxLat, selectedArea.maxLon]]}
                  pathOptions={{
                    color: '#00ff00',
                    weight: 2,
                    fillColor: '#00ff00',
                    fillOpacity: 0.1
                  }}
                />
              )}
            </MapContainer>
            
            {/* Loading indicator */}
            {(buildingsLoading || roadsLoading || waterLoading || greenLoading) && (
              <div className="absolute top-4 right-4 bg-gray-800/95 backdrop-blur-sm rounded-lg p-2 border border-gray-700 shadow-lg">
                <div className="text-sm text-gray-300">Loading OSM data...</div>
              </div>
            )}
          </div>
        ) : (
          <div className="w-full h-full flex items-center justify-center bg-gray-800">
            <div className="text-center">
              <div className="text-6xl mb-4"></div>
              <p className="text-gray-400 text-xl">Select a city from the panel to view map</p>
            </div>
          </div>
        )}
        
        {/* Map Controls Overlay */}
        <div className="absolute top-4 left-4 bg-gray-800/95 backdrop-blur-sm rounded-lg p-3 border border-gray-700 shadow-lg z-10">
          <div className="text-sm font-semibold text-white mb-3">Layer Toggle</div>
          <div className="space-y-2">
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showBuildings}
                onChange={(e) => setShowBuildings(e.target.checked)}
                className="w-4 h-4 text-orange-600 bg-gray-700 border-gray-600 rounded"
              />
              <span className="text-sm text-white">Buildings</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showRoads}
                onChange={(e) => setShowRoads(e.target.checked)}
                className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
              />
              <span className="text-sm text-white">Roads</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showWater}
                onChange={(e) => setShowWater(e.target.checked)}
                className="w-4 h-4 text-cyan-600 bg-gray-700 border-gray-600 rounded"
              />
              <span className="text-sm text-white">Water</span>
            </label>
            <label className="flex items-center space-x-2 cursor-pointer">
              <input
                type="checkbox"
                checked={showGreen}
                onChange={(e) => setShowGreen(e.target.checked)}
                className="w-4 h-4 text-green-600 bg-gray-700 border-gray-600 rounded"
              />
              <span className="text-sm text-white">Green</span>
            </label>
          </div>
          <div className="mt-3 pt-3 border-t border-gray-700 space-y-2">
            <label className="block text-xs text-gray-400 mb-1">Scenario</label>
            <input
              type="text"
              value={scenario}
              onChange={(e) => setScenario(e.target.value)}
              placeholder="baseline"
              className="w-full px-2 py-1 bg-gray-700 border border-gray-600 rounded text-sm text-white"
            />
            <div className="space-y-2 mt-2">
              <button
                onClick={() => setIsSelectingArea(!isSelectingArea)}
                className={`w-full px-2 py-1 text-xs rounded transition-colors ${
                  isSelectingArea 
                    ? 'bg-red-600 hover:bg-red-700 text-white' 
                    : 'bg-gray-700 hover:bg-gray-600 text-white'
                }`}
              >
                {isSelectingArea ? 'Cancel Selection' : 'Select Area'}
              </button>
              {selectedArea && (
                <>
                  <label className="flex items-center space-x-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={filterByArea}
                      onChange={(e) => setFilterByArea(e.target.checked)}
                      className="w-4 h-4 text-green-600 bg-gray-700 border-gray-600 rounded"
                    />
                    <span className="text-xs text-white">Filter by Area</span>
                  </label>
                  <button
                    onClick={() => {
                      setSelectedArea(null)
                      setFilterByArea(false)
                    }}
                    className="w-full px-2 py-1 text-xs bg-gray-600 hover:bg-gray-700 rounded text-white"
                  >
                    Clear Area
                  </button>
                </>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Right Settings Panel */}
      <div className={`bg-gray-800 border-l border-gray-700 transition-all duration-300 ${isPanelOpen ? 'w-96' : 'w-0'} overflow-hidden`}>
        <div className="h-full flex flex-col">
          {/* Panel Header */}
          <div className="p-4 border-b border-gray-700 flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Control Panel</h2>
            <button
              onClick={() => setIsPanelOpen(false)}
              className="text-gray-400 hover:text-white"
            >
              Close
            </button>
          </div>

          {/* City Selector */}
          <div className="p-4 border-b border-gray-700">
            <label className="block text-sm font-medium mb-2 text-gray-300">
              Select City
            </label>
            <select
              value={selectedCityId || ''}
              onChange={(e) => setSelectedCityId(Number(e.target.value))}
              className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">-- Select a city --</option>
              {cities?.map((city) => (
                <option key={city.id} value={city.id}>
                  {city.name}, {city.country}
                </option>
              ))}
            </select>
          </div>

          {/* Feature Tabs */}
          <div className="border-b border-gray-700">
            <div className="flex overflow-x-auto">
              {[
                { id: 'weather', label: 'Weather', name: 'weather' },
                { id: 'traffic', label: 'Traffic', name: 'traffic' },
                { id: 'urban', label: 'Urban', name: 'urban' },
                { id: 'water', label: 'Water', name: 'water' },
                { id: 'buildings', label: 'Buildings', name: 'buildings' },
                { id: 'roads', label: 'Roads', name: 'roads' },
                { id: 'green', label: 'Green', name: 'green' }
              ].map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.name as any)}
                  className={`px-4 py-3 text-sm font-medium border-b-2 transition-colors whitespace-nowrap ${
                    activeTab === tab.name
                      ? 'border-blue-500 text-blue-400'
                      : 'border-transparent text-gray-400 hover:text-gray-300'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          </div>

          {/* Scrollable Content Area */}
          <div className="flex-1 overflow-y-auto p-4">
            {activeTab === 'weather' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white mb-3">Weather Prediction & Simulation</h3>
                
                {latestClimate && (
                  <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                    <div className="text-3xl font-bold text-white mb-1">
                      {latestClimate.temperature.toFixed(1)}°C
                    </div>
                    <div className="text-sm text-gray-400">
                      Humidity: {latestClimate.humidity?.toFixed(1) || 'N/A'}%
                    </div>
                  </div>
                )}

                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Prediction Timeframe</label>
                    <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white">
                      <option>Next 24 Hours</option>
                      <option>Next 7 Days</option>
                      <option>Next 30 Days</option>
                      <option>Next 3 Months</option>
                    </select>
                  </div>

                  <button className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white font-medium transition-colors">
                    Run Weather Simulation
                  </button>

                  <div className="bg-gray-700/50 rounded p-3 text-sm text-gray-300">
                    <div className="font-semibold mb-2">Climate Factors:</div>
                    <div className="space-y-1">
                      <div>• Temperature trends</div>
                      <div>• Precipitation patterns</div>
                      <div>• Wind speed & direction</div>
                      <div>• Air quality index</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'traffic' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white mb-3">Traffic Prediction & Simulation</h3>
                
                {latestTraffic && (
                  <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                    <div className="text-2xl font-bold text-white mb-1">
                      {latestTraffic.congestion_level || 'N/A'}
                    </div>
                    <div className="text-sm text-gray-400">
                      Speed: {latestTraffic.speed?.toFixed(1) || 'N/A'} km/h
                    </div>
                  </div>
                )}

                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Simulation Time</label>
                    <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white">
                      <option>Peak Hours (7-9 AM)</option>
                      <option>Midday (12-2 PM)</option>
                      <option>Evening (5-7 PM)</option>
                      <option>Night (10 PM-6 AM)</option>
                    </select>
                  </div>

                  <button className="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 rounded text-white font-medium transition-colors">
                    Simulate Traffic Flow
                  </button>

                  <div className="bg-gray-700/50 rounded p-3 text-sm text-gray-300">
                    <div className="font-semibold mb-2">Analysis Metrics:</div>
                    <div className="space-y-1">
                      <div>• Congestion levels</div>
                      <div>• Vehicle density</div>
                      <div>• Average speed</div>
                      <div>• Accident hotspots</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'urban' && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold text-white mb-3">Urban Growth & Economy</h3>
                
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Projection Period</label>
                    <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white">
                      <option>1 Year</option>
                      <option>5 Years</option>
                      <option>10 Years</option>
                      <option>20 Years</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm text-gray-400 mb-2">Growth Rate (%)</label>
                    <input
                      type="number"
                      step="0.1"
                      placeholder="3.5"
                      className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                    />
                  </div>

                  <button className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded text-white font-medium transition-colors">
                    Calculate Projections
                  </button>

                  <div className="bg-gray-700/50 rounded p-3 text-sm text-gray-300">
                    <div className="font-semibold mb-2">Mathematical Models:</div>
                    <div className="space-y-1">
                      <div>• Population: P(t) = P₀(1+r)ᵗ</div>
                      <div>• GDP Growth: Y = C + I + G + NX</div>
                      <div>• Urban Density: ρ = M/A</div>
                      <div>• Infrastructure Load</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {activeTab === 'water' && (
              <OSMCRUD 
                layerType="water" 
                onFeatureSelect={handleFeatureSelect}
                onFeatureHighlight={handleFeatureHighlight}
              />
            )}

            {activeTab === 'roads' && (
              <OSMCRUD 
                layerType="roads" 
                onFeatureSelect={handleFeatureSelect}
                onFeatureHighlight={handleFeatureHighlight}
              />
            )}

            {activeTab === 'green' && (
              <OSMCRUD 
                layerType="green" 
                onFeatureSelect={handleFeatureSelect}
                onFeatureHighlight={handleFeatureHighlight}
              />
            )}

            {activeTab === 'buildings' && (
              <OSMCRUD 
                layerType="buildings" 
                onFeatureSelect={handleFeatureSelect}
                onFeatureHighlight={handleFeatureHighlight}
              />
            )}
          </div>
        </div>
      </div>

      {/* Toggle Panel Button (when closed) */}
      {!isPanelOpen && (
        <button
          onClick={() => setIsPanelOpen(true)}
          className="fixed top-4 right-4 px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-white shadow-lg transition-colors"
        >
          Open Panel
        </button>
      )}
    </div>
  )
}

