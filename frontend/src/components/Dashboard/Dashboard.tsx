import { useState, useEffect, useRef } from 'react'
import ErrorBoundary from './ErrorBoundary'
import { useOSMByBBox } from '../../hooks/useOSM'
import { PHYS } from '../../constants/phys'
import { useWeather } from '../../hooks/useWeather'
import { Viewer, GeoJsonDataSource, Entity, PolygonGraphics } from 'resium'
import { Cartesian3, Color, CallbackProperty, ConstantProperty, ScreenSpaceEventHandler, ScreenSpaceEventType, defined } from 'cesium'
import OSMCRUD from './OSMCRUD'

// Component to handle Cesium camera bounds changes and fetch OSM data
function CesiumBoundsHandler({
  onBoundsChange,
  viewerRef
}: {
  onBoundsChange: (bbox: { minLon: number; minLat: number; maxLon: number; maxLat: number }) => void
  viewerRef: React.MutableRefObject<any>
}) {
  const timeoutRef = useRef<NodeJS.Timeout | number>()

  useEffect(() => {
    if (!viewerRef.current) return

    const updateBounds = () => {
      const viewer = viewerRef.current.cesiumElement
      if (!viewer) return

      const scene = viewer.scene
      const ellipsoid = scene.globe.ellipsoid
      const camera = viewer.camera

      // Get camera view rectangle
      const viewRectangle = camera.computeViewRectangle(ellipsoid)
      if (!defined(viewRectangle)) return

      const bbox = {
        minLon: viewRectangle.west * (180 / Math.PI),
        minLat: viewRectangle.south * (180 / Math.PI),
        maxLon: viewRectangle.east * (180 / Math.PI),
        maxLat: viewRectangle.north * (180 / Math.PI),
      }
      onBoundsChange(bbox)
    }

    // Initial bounds
    updateBounds()

    // Update bounds on camera change (debounced)
    const handleCameraChange = () => {
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      timeoutRef.current = setTimeout(updateBounds, 500)
    }

    const viewer = viewerRef.current.cesiumElement
    viewer.camera.changed.addEventListener(handleCameraChange)

    return () => {
      if (viewer) {
        viewer.camera.changed.removeEventListener(handleCameraChange)
      }
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
    }
  }, [onBoundsChange, viewerRef])

  return null
}

// Component for area selection (rectangle drawing) in Cesium
function CesiumAreaSelector({
  isSelecting,
  onSelectionComplete,
  viewerRef
}: {
  isSelecting: boolean
  onSelectionComplete: (bbox: { minLon: number; minLat: number; maxLon: number; maxLat: number }) => void
  viewerRef: React.MutableRefObject<any>
}) {
  const [startPoint, setStartPoint] = useState<Cartesian3 | null>(null)
  const [endPoint, setEndPoint] = useState<Cartesian3 | null>(null)

  useEffect(() => {
    if (!viewerRef.current || !isSelecting) return

    const viewer = viewerRef.current.cesiumElement
    if (!viewer) return

    const handler = new ScreenSpaceEventHandler(viewer.scene.canvas)

    handler.setInputAction((click: any) => {
      const cartesian = viewer.camera.pickEllipsoid(click.position, viewer.scene.globe.ellipsoid)
      if (!cartesian) return

      if (!startPoint) {
        setStartPoint(cartesian)
      } else {
        setEndPoint(cartesian)

        // Convert to lat/lon
        const startCartographic = viewer.scene.globe.ellipsoid.cartesianToCartographic(startPoint)
        const endCartographic = viewer.scene.globe.ellipsoid.cartesianToCartographic(cartesian)

        const minLat = Math.min(startCartographic.latitude, endCartographic.latitude) * (180 / Math.PI)
        const maxLat = Math.max(startCartographic.latitude, endCartographic.latitude) * (180 / Math.PI)
        const minLon = Math.min(startCartographic.longitude, endCartographic.longitude) * (180 / Math.PI)
        const maxLon = Math.max(startCartographic.longitude, endCartographic.longitude) * (180 / Math.PI)

        onSelectionComplete({ minLon, minLat, maxLon, maxLat })
        setStartPoint(null)
        setEndPoint(null)
      }
    }, ScreenSpaceEventType.LEFT_CLICK)

    return () => {
      handler.destroy()
    }
  }, [isSelecting, startPoint, onSelectionComplete, viewerRef])

  if (!isSelecting || !startPoint || !endPoint) return null

  // Create rectangle entity for selection preview
  const startCartographic = viewerRef.current?.cesiumElement?.scene?.globe?.ellipsoid?.cartesianToCartographic(startPoint)
  const endCartographic = viewerRef.current?.cesiumElement?.scene?.globe?.ellipsoid?.cartesianToCartographic(endPoint)

  if (!startCartographic || !endCartographic) return null

  const minLat = Math.min(startCartographic.latitude, endCartographic.latitude) * (180 / Math.PI)
  const maxLat = Math.max(startCartographic.latitude, endCartographic.latitude) * (180 / Math.PI)
  const minLon = Math.min(startCartographic.longitude, endCartographic.longitude) * (180 / Math.PI)
  const maxLon = Math.max(startCartographic.longitude, endCartographic.longitude) * (180 / Math.PI)

  const rectangleHierarchy = new CallbackProperty(() => {
    return {
      positions: Cartesian3.fromDegreesArray([
        minLon, minLat,
        maxLon, minLat,
        maxLon, maxLat,
        minLon, maxLat
      ])
    }
  }, false)

  return (
    <Entity>
      <PolygonGraphics
        hierarchy={rectangleHierarchy}
        material={Color.RED.withAlpha(0.2)}
        outline={true}
        outlineColor={Color.RED}
        outlineWidth={2}
      />
    </Entity>
  )
}

export default function Dashboard() {
    // Scenario parameters for physical meaning
    const scenarioParams: Record<string, { treeFactor: number; heatFactor: number }> = {
      baseline: { treeFactor: 1, heatFactor: 1 },
      green: { treeFactor: 1.2, heatFactor: 0.85 },
      heatwave: { treeFactor: 0.9, heatFactor: 1.2 },
    }
  const [activeTab, setActiveTab] = useState<'weather' | 'traffic' | 'urban' | 'water' | 'buildings' | 'roads' | 'green'>('weather')
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [mapBounds, setMapBounds] = useState<{ minLon: number; minLat: number; maxLon: number; maxLat: number } | null>(null)
  const [scenario, setScenario] = useState<string>('baseline')
  const [showBuildings, setShowBuildings] = useState(true)
  const [showRoads, setShowRoads] = useState(false)
  const [showWater, setShowWater] = useState(false)
  const [showGreen, setShowGreen] = useState(false)
  const [highlightedOsmId, setHighlightedOsmId] = useState<number | null>(null)
  const [selectedArea, setSelectedArea] = useState<{ minLon: number; minLat: number; maxLon: number; maxLat: number } | null>(null)
  const [isSelectingArea, setIsSelectingArea] = useState(false)
  const [filterByArea, setFilterByArea] = useState(false)

  const viewerRef = useRef<any>(null)

  // Hardcoded Bengaluru city
  const bengaluru = { name: 'Bengaluru', country: 'India', latitude: 12.9716, longitude: 77.5946 }
  // Mock trafficData for Bengaluru (replace with real hook if needed)
  const trafficData = [{ congestion_level: 'Moderate', speed: 25.0 }];
  const { data: weatherData, isLoading: weatherLoading } = useWeather(bengaluru.latitude, bengaluru.longitude)

  // Determine which bbox to use (selected area or map bounds)
  const activeBbox = filterByArea && selectedArea ? selectedArea : mapBounds

  // Fetch OSM data from backend based on map bounds or selected area
  // Debugging: log bbox and overlay fetches
  useEffect(() => {
    if (activeBbox) {
      console.debug('[Cesium] Active bbox:', activeBbox)
    }
  }, [activeBbox])

  const { data: buildingsData, isLoading: buildingsLoading, error: buildingsError } = useOSMByBBox(
    'buildings',
    activeBbox,
    { scenario, active: true, enabled: showBuildings && !!activeBbox }
  )
  useEffect(() => {
    if (buildingsError) console.error('[OSM] Buildings fetch error:', buildingsError)
    if (buildingsData) console.debug('[OSM] Buildings overlay loaded:', buildingsData)
  }, [buildingsData, buildingsError])

  const { data: roadsData, isLoading: roadsLoading, error: roadsError } = useOSMByBBox(
    'roads',
    activeBbox,
    { scenario, active: true, enabled: showRoads && !!activeBbox }
  )
  useEffect(() => {
    if (roadsError) console.error('[OSM] Roads fetch error:', roadsError)
    if (roadsData) console.debug('[OSM] Roads overlay loaded:', roadsData)
  }, [roadsData, roadsError])

  const { data: waterData, isLoading: waterLoading, error: waterError } = useOSMByBBox(
    'water',
    activeBbox,
    { scenario, active: true, enabled: showWater && !!activeBbox }
  )
  useEffect(() => {
    if (waterError) console.error('[OSM] Water fetch error:', waterError)
    if (waterData) console.debug('[OSM] Water overlay loaded:', waterData)
  }, [waterData, waterError])

  const { data: greenData, isLoading: greenLoading, error: greenError } = useOSMByBBox(
    'green',
    activeBbox,
    { scenario, active: true, enabled: showGreen && !!activeBbox }
  )
  useEffect(() => {
    if (greenError) console.error('[OSM] Green fetch error:', greenError)
    if (greenData) console.debug('[OSM] Green overlay loaded:', greenData)
  }, [greenData, greenError])

  // Handle feature selection from OSMCRUD
  const handleFeatureSelect = (feature: any) => {
    const osmId = feature.properties?.osm_id || feature.properties?.id
    if (osmId) {
      setHighlightedOsmId(osmId)
    }
  }

  const handleFeatureHighlight = (osmId: number | null) => {
    setHighlightedOsmId(osmId)
  }

  const handleAreaSelection = (bbox: { minLon: number; minLat: number; maxLon: number; maxLat: number }) => {
    setSelectedArea(bbox)
    setIsSelectingArea(false)
    setFilterByArea(true)
  }

  // Removed unused latestClimate
  const latestTraffic = trafficData?.[0]
  // Removed duplicate selectedCity

  // Set initial map bounds when city is selected
  useEffect(() => {
    // Set initial bounds around Bengaluru (approximate 10km radius)
    const lat = bengaluru.latitude
    const lon = bengaluru.longitude
    const offset = 0.05 // ~5km
    setMapBounds({
      minLon: lon - offset,
      minLat: lat - offset,
      maxLon: lon + offset,
      maxLat: lat + offset,
    })
    console.debug('[Cesium] Initial city bounds set:', {
      minLon: lon - offset,
      minLat: lat - offset,
      maxLon: lon + offset,
      maxLat: lat + offset,
    })
  }, [])

  // Maintain per-layer Cesium DataSource refs
  const dataSourcesRef = useRef<Record<string, any>>({
    buildings: null,
    roads: null,
    water: null,
    green: null,
  })

  // Helper to add/update Cesium DataSource for a layer
  const handleLayerLoad = (name: string, ds: any, viewer: any) => {
    if (dataSourcesRef.current[name]) {
      viewer.dataSources.remove(dataSourcesRef.current[name])
    }
    dataSourcesRef.current[name] = ds
    viewer.dataSources.add(ds)
  }

  // Zoom-aware layer loading
  const [cameraHeight, setCameraHeight] = useState<number>(1000000)
  useEffect(() => {
    const viewer = viewerRef.current?.cesiumElement
    if (!viewer) return
    const updateHeight = () => {
      const height = viewer.camera.positionCartographic.height
      setCameraHeight(height)
      console.debug('[Cesium] Camera height:', height)
    }
    updateHeight()
    viewer.camera.changed.addEventListener(updateHeight)
    return () => {
      viewer.camera.changed.removeEventListener(updateHeight)
    }
  }, [viewerRef])

  // Layer load rules
  const canLoadBuildings = cameraHeight < 50000
  // Removed unused canLoadRoads
  // Removed unused canLoadGreen
  // Removed unused canLoadWater

  return (
    <div className="fixed inset-0 flex bg-gray-900">
      {/* Full-screen OSM Buildings & Roads Visualization */}
      <div className="flex-1 relative">
        <div className="w-full h-full bg-gray-900">
          <ErrorBoundary>
            {/* ...existing code for <Viewer> and its children... */}
            <Viewer
              ref={viewerRef}
              style={{ height: '100%', width: '100%' }}
              key={`viewer-bengaluru`}
              baseLayerPicker={false}
              geocoder={false}
              homeButton={false}
              sceneModePicker={false}
              navigationHelpButton={false}
              animation={false}
              timeline={false}
              fullscreenButton={false}
              vrButton={false}
            >
              {/* ...existing code for CesiumBoundsHandler, CesiumAreaSelector, and layers... */}
              <CesiumBoundsHandler onBoundsChange={setMapBounds} viewerRef={viewerRef} />
              <CesiumAreaSelector
                isSelecting={isSelectingArea}
                onSelectionComplete={handleAreaSelection}
                viewerRef={viewerRef}
              />
              {/* ...existing code for layers and overlays... */}
              {/* Water Layer (Base) */}
              {showWater && waterData && (
                <GeoJsonDataSource
                  data={waterData}
                  show={true}
                  onLoad={(dataSource) => {
                    const viewer = viewerRef.current?.cesiumElement
                    if (viewer) handleLayerLoad('water', dataSource, viewer)
                    const entities = dataSource.entities.values
                    entities.forEach((entity) => {
                      if (entity.polygon) {
                        const osmId = entity.properties?.osm_id?.getValue()
                        const isHighlighted = osmId === highlightedOsmId
                        const materialProp = new CallbackProperty(() =>
                          Color.fromCssColorString(isHighlighted ? '#ff0000' : '#03a9f4').withAlpha(isHighlighted ? 0.8 : 0.5), false) as any;
                        (materialProp as any).getType = () => 'Color';
                        entity.polygon.material = materialProp;
                        entity.polygon.outlineColor = new CallbackProperty(() =>
                          Color.fromCssColorString(isHighlighted ? '#ff0000' : '#03a9f4'), false)
                        entity.polygon.outlineWidth = new CallbackProperty(() =>
                          isHighlighted ? 3 : 1, false)
                      }
                    })
                  }}
                />
              )}
              {/* Green Spaces Layer */}
              {showGreen && greenData && (
                <GeoJsonDataSource
                  data={greenData}
                  show={true}
                  onLoad={(dataSource) => {
                    const viewer = viewerRef.current?.cesiumElement
                    if (viewer) handleLayerLoad('green', dataSource, viewer)
                    const entities = dataSource.entities.values
                    entities.forEach((entity) => {
                      if (entity.polygon) {
                        const osmId = entity.properties?.osm_id?.getValue()
                        const isHighlighted = osmId === highlightedOsmId
                        const materialProp = new CallbackProperty(() =>
                          Color.fromCssColorString(isHighlighted ? '#ff0000' : '#4caf50').withAlpha(
                            (isHighlighted ? 0.8 : 0.4) * (scenarioParams[scenario]?.treeFactor ?? 1)
                          ), false) as any;
                        (materialProp as any).getType = () => 'Color';
                        entity.polygon.material = materialProp;
                        entity.polygon.outlineColor = new CallbackProperty(() =>
                          Color.fromCssColorString(isHighlighted ? '#ff0000' : '#4caf50'), false)
                        entity.polygon.outlineWidth = new CallbackProperty(() =>
                          isHighlighted ? 3 : 1, false)
                      }
                    })
                  }}
                />
              )}
              {/* Roads Layer */}
              {showRoads && roadsData && (
                <GeoJsonDataSource
                  data={roadsData}
                  show={true}
                  onLoad={(dataSource) => {
                    const viewer = viewerRef.current?.cesiumElement
                    if (viewer) handleLayerLoad('roads', dataSource, viewer)
                    const entities = dataSource.entities.values
                    entities.forEach((entity) => {
                      if (entity.polyline) {
                        const osmId = entity.properties?.osm_id?.getValue()
                        const isHighlighted = osmId === highlightedOsmId
                        const materialProp = new CallbackProperty(() =>
                          Color.fromCssColorString(isHighlighted ? '#ff0000' : '#2196f3').withAlpha(isHighlighted ? 1 : 0.8), false) as any;
                        (materialProp as any).getType = () => 'Color';
                        entity.polyline.material = materialProp;
                        entity.polyline.width = new CallbackProperty(() =>
                          isHighlighted ? 4 : 2, false)
                      }
                    })
                  }}
                />
              )}
              {/* Buildings Layer (Top) */}
              {showBuildings && canLoadBuildings && buildingsData && (
                <GeoJsonDataSource
                  data={buildingsData}
                  show={true}
                  onLoad={(dataSource) => {
                    const viewer = viewerRef.current?.cesiumElement
                    if (viewer) handleLayerLoad('buildings', dataSource, viewer)
                    const entities = dataSource.entities.values
                    entities.forEach((entity) => {
                      if (entity.polygon) {
                        const osmId = entity.properties?.osm_id?.getValue() ?? 1;
                        const isHighlighted = osmId === highlightedOsmId;
                        // Use OSM building_levels if available, else fallback
                        const levels = entity.properties?.building_levels?.getValue?.();
                        const height = levels ? levels * PHYS.FLOOR_HEIGHT_M : PHYS.DEFAULT_BUILDING_HEIGHT_M;
                        entity.polygon.extrudedHeight = new ConstantProperty(height);
                        // Fix CallbackProperty getType error for material
                        const materialProp = new CallbackProperty(() =>
                          Color.fromCssColorString(isHighlighted ? '#ff0000' : '#ff9800').withAlpha(isHighlighted ? 0.8 : 0.4), false) as any;
                        (materialProp as any).getType = () => 'Color';
                        entity.polygon.material = materialProp;
                        entity.polygon.outlineColor = new CallbackProperty(() =>
                          Color.fromCssColorString(isHighlighted ? '#ff0000' : '#ff9800'), false);
                        entity.polygon.outlineWidth = new CallbackProperty(() =>
                          isHighlighted ? 3 : 1, false);
                      }
                    })
                  }}
                />
              )}
              {/* Selected Area Rectangle */}
              {selectedArea && filterByArea && (
                <Entity>
                  <PolygonGraphics
                    hierarchy={Cartesian3.fromDegreesArray([
                      selectedArea.minLon, selectedArea.minLat,
                      selectedArea.maxLon, selectedArea.minLat,
                      selectedArea.maxLon, selectedArea.maxLat,
                      selectedArea.minLon, selectedArea.maxLat
                    ])}
                    material={Color.fromCssColorString('#00ff00').withAlpha(0.1)}
                    outline={true}
                    outlineColor={Color.fromCssColorString('#00ff00')}
                    outlineWidth={2}
                  />
                </Entity>
              )}
              {/* ...existing code for overlays... */}
            </Viewer>
          </ErrorBoundary>
            
            {/* Loading indicator */}
            {(buildingsLoading || roadsLoading || waterLoading || greenLoading) && (
              <div className="absolute top-4 right-4 bg-gray-800/95 backdrop-blur-sm rounded-lg p-2 border border-gray-700 shadow-lg">
                <div className="text-sm text-gray-300">Loading OSM data...</div>
              </div>
            )}
          </div>
        {/* No city selector fallback needed, always Bengaluru */}
        
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
          {/* City Selector removed, always Bengaluru */}

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
                <h3 className="text-lg font-semibold text-white mb-3">Current Weather</h3>

                {weatherLoading ? (
                  <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                    <div className="text-sm text-gray-400">Loading weather data...</div>
                  </div>
                ) : weatherData ? (
                  <div className="space-y-4">
                    {/* Current Weather */}
                    <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                      <div className="flex items-center justify-between mb-2">
                        <div className="text-lg font-semibold text-white">
                          {weatherData.location.name}, {weatherData.location.country}
                        </div>
                        <div className="text-sm text-gray-400">
                          {new Date(weatherData.current.last_updated).toLocaleString()}
                        </div>
                      </div>
                      <div className="flex items-center space-x-4">
                        <img
                          src={weatherData.current.condition.icon}
                          alt={weatherData.current.condition.text}
                          className="w-16 h-16"
                        />
                        <div>
                          <div className="text-3xl font-bold text-white mb-1">
                            {weatherData.current.temp_c}°C
                          </div>
                          <div className="text-sm text-gray-400">
                            Feels like {weatherData.current.feelslike_c}°C
                          </div>
                          <div className="text-sm text-gray-300">
                            {weatherData.current.condition.text}
                          </div>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 gap-4 mt-4 text-sm">
                        <div>
                          <div className="text-gray-400">Humidity</div>
                          <div className="text-white">{weatherData.current.humidity}% RH</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Wind</div>
                          <div className="text-white">{weatherData.current.wind_kph} km/h {weatherData.current.wind_dir}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Pressure</div>
                          <div className="text-white">{weatherData.current.pressure_mb} mb</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Visibility</div>
                          <div className="text-white">{weatherData.current.vis_km} km</div>
                        </div>
                      </div>
                    </div>

                    {/* Forecast Preview */}
                    <div className="bg-gray-700/50 rounded p-3">
                      <div className="font-semibold mb-2 text-white">7-Day Forecast</div>
                      <div className="text-sm text-gray-300">
                        Forecast data available - select timeframe to view detailed predictions
                      </div>
                    </div>

                    <div className="space-y-3">
                      <div>
                        <label className="block text-sm text-gray-400 mb-2">View Forecast</label>
                        <select className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white">
                          <option>Next 24 Hours</option>
                          <option>Next 7 Days</option>
                          <option>Next 14 Days</option>
                        </select>
                      </div>

                      <button className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded text-white font-medium transition-colors">
                        Load Forecast Data
                      </button>

                      <div className="bg-gray-700/50 rounded p-3 text-sm text-gray-300">
                        <div className="font-semibold mb-2">Weather Factors:</div>
                        <div className="space-y-1">
                          <div>• Temperature variations</div>
                          <div>• Precipitation forecasts</div>
                          <div>• Wind patterns</div>
                          <div>• Air quality monitoring</div>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="bg-gray-700 rounded-lg p-4 border border-gray-600">
                    <div className="text-sm text-gray-400">No weather data available</div>
                  </div>
                )}
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
                    {/* Units: km/h, congestion_level: qualitative */}
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

