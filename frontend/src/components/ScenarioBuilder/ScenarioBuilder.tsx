import { Suspense, useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { useCityBuildings, useCityClimate } from '../../hooks/useCities'
import BuildingMesh from './BuildingMesh'
import { ClimateData } from '../../types/climate'

// API Configuration
const API_BASE_URL = 'http://localhost:8000/api'

// API Functions
const citiesApi = {
  getAll: async () => {
    const response = await fetch(`${API_BASE_URL}/cities/`)
    const data = await response.json()
    if (Array.isArray(data)) return data
    if (Array.isArray(data?.results)) return data.results
    return []
  },
}

const scenariosApi = {
  getAll: async () => {
    const response = await fetch(`${API_BASE_URL}/scenarios/`)
    const data = await response.json()
    if (Array.isArray(data)) return data
    if (Array.isArray(data?.results)) return data.results
    return []
  },
  create: async (data: any) => {
    const response = await fetch(`${API_BASE_URL}/scenarios/`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    })
    if (!response.ok) {
      throw new Error('Failed to create scenario')
    }
    return response.json()
  },
  delete: async (id: number) => {
    const response = await fetch(`${API_BASE_URL}/scenarios/${id}/`, {
      method: 'DELETE',
    })
    if (!response.ok) {
      throw new Error('Failed to delete scenario')
    }
    return true
  },
}

// Hooks
const useCities = () => {
  return useQuery({
    queryKey: ['cities'],
    queryFn: () => citiesApi.getAll(),
  })
}

const useScenarios = () => {
  return useQuery({
    queryKey: ['scenarios'],
    queryFn: () => scenariosApi.getAll(),
  })
}

export default function ScenarioBuilder() {
  const [selectedCityId, setSelectedCityId] = useState<number | null>(null)
  const [scenarioName, setScenarioName] = useState('')
  const [description, setDescription] = useState('')
  const [timeHorizon, setTimeHorizon] = useState('1y')
  const [vegetationChange, setVegetationChange] = useState('0')
  const [buildingDensityChange, setBuildingDensityChange] = useState('0')
  const [message, setMessage] = useState<any>(null)
  const [isPanelOpen, setIsPanelOpen] = useState(true)
  const [viewMode, setViewMode] = useState<'map' | '3d'>('map')
  const [overlayType, setOverlayType] = useState<'temperature' | 'humidity' | 'precipitation'>('temperature')

  const { data: cities } = useCities()
  const { data: scenarios } = useScenarios()
  const { data: buildingsGeoJSON } = useCityBuildings(selectedCityId || 0)
  const { data: climateData } = useCityClimate(selectedCityId || 0)
  const queryClient = useQueryClient()

  const latestClimate = climateData?.[0]

  const createScenarioMutation = useMutation({
    mutationFn: scenariosApi.create,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
      setScenarioName('')
      setDescription('')
      setVegetationChange('0')
      setBuildingDensityChange('0')
      setSelectedCityId(null)
      setMessage({ type: 'success', text: 'Scenario created successfully!' })
      setTimeout(() => setMessage(null), 3000)
    },
    onError: (error) => {
      setMessage({ type: 'error', text: `Error: ${error.message}` })
      setTimeout(() => setMessage(null), 5000)
    },
  })

  const deleteScenarioMutation = useMutation({
    mutationFn: scenariosApi.delete,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
      setMessage({ type: 'success', text: 'Scenario deleted successfully!' })
      setTimeout(() => setMessage(null), 3000)
    },
    onError: (error) => {
      setMessage({ type: 'error', text: `Error: ${error.message}` })
      setTimeout(() => setMessage(null), 5000)
    },
  })

  const handleSubmit = () => {
    if (!selectedCityId) {
      setMessage({ type: 'error', text: 'Please select a city' })
      setTimeout(() => setMessage(null), 3000)
      return
    }

    if (!scenarioName.trim()) {
      setMessage({ type: 'error', text: 'Please enter a scenario name' })
      setTimeout(() => setMessage(null), 3000)
      return
    }

    createScenarioMutation.mutate({
      name: scenarioName,
      description,
      city: selectedCityId,
      time_horizon: timeHorizon,
      parameters: {
        vegetation_change: parseFloat(vegetationChange),
        building_density_change: parseFloat(buildingDensityChange),
      },
    })
  }

  const handleDeleteScenario = (id: number) => {
    if (window.confirm('Are you sure you want to delete this scenario?')) {
      deleteScenarioMutation.mutate(id)
    }
  }

  const selectedCity = cities?.find((city: any) => city.id === selectedCityId)

  return (
    <div className="fixed inset-0 flex bg-gray-900">
      {/* Full-screen Viewer (Map or 3D) */}
      <div className="flex-1 relative">
        {/* View Mode Toggle */}
        <div className="absolute top-4 left-4 z-10 flex gap-2">
          <button
            onClick={() => setViewMode('map')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === 'map'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            🗺️ Map View
          </button>
          <button
            onClick={() => setViewMode('3d')}
            className={`px-4 py-2 rounded-lg font-medium transition-colors ${
              viewMode === '3d'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
            }`}
          >
            🏙️ 3D View
          </button>
          {viewMode === '3d' && selectedCityId && (
            <select
              value={overlayType}
              onChange={(e) => setOverlayType(e.target.value as any)}
              className="px-3 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm"
            >
              <option value="temperature">🌡️ Temperature</option>
              <option value="humidity">💧 Humidity</option>
              <option value="precipitation">🌧️ Precipitation</option>
            </select>
          )}
          {latestClimate && viewMode === '3d' && (
            <div className="px-4 py-2 bg-gray-800 border border-gray-700 rounded-lg text-white text-sm">
              {overlayType === 'temperature' && `${latestClimate.temperature.toFixed(1)}°C`}
              {overlayType === 'humidity' && `${latestClimate.humidity?.toFixed(1) || 'N/A'}%`}
              {overlayType === 'precipitation' && `${latestClimate.precipitation?.toFixed(1) || 'N/A'}mm`}
            </div>
          )}
        </div>

        {/* Map View */}
        {viewMode === 'map' && (
          <>
            {selectedCity ? (
              <iframe
                src={`https://www.openstreetmap.org/export/embed.html?bbox=${selectedCity.longitude - 0.1},${selectedCity.latitude - 0.1},${selectedCity.longitude + 0.1},${selectedCity.latitude + 0.1}&layer=mapnik&marker=${selectedCity.latitude},${selectedCity.longitude}`}
                className="w-full h-full"
                style={{ border: 0 }}
                title="Scenario Map"
              />
            ) : (
              <div className="w-full h-full flex items-center justify-center bg-gray-800">
                <div className="text-center">
                  <div className="text-6xl mb-4">🎯</div>
                  <p className="text-gray-400 text-xl">Select a city to build scenarios</p>
                </div>
              </div>
            )}
          </>
        )}

        {/* 3D View */}
        {viewMode === '3d' && (
          <div className="w-full h-full bg-gray-900">
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
                {selectedCityId ? 'Loading 3D buildings...' : 'Select a city to view 3D buildings'}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Right Panel */}
      <div className={`bg-gray-800 border-l border-gray-700 transition-all duration-300 ${isPanelOpen ? 'w-96' : 'w-0'} overflow-hidden`}>
        <div className="h-full flex flex-col">
          <div className="p-4 border-b border-gray-700 flex items-center justify-between">
            <h2 className="text-xl font-bold text-white">Scenario Builder</h2>
            <button
              onClick={() => setIsPanelOpen(false)}
              className="text-gray-400 hover:text-white"
            >
              ✕
            </button>
          </div>

          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            {message && (
              <div className={`border rounded-lg p-3 text-sm ${
                message.type === 'success' 
                  ? 'bg-green-900/20 border-green-700 text-green-400' 
                  : 'bg-red-900/20 border-red-700 text-red-400'
              }`}>
                {message.text}
              </div>
            )}

            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">Select City *</label>
              <select
                value={selectedCityId || ''}
                onChange={(e) => setSelectedCityId(Number(e.target.value))}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
              >
                <option value="">-- Select a city --</option>
                {cities?.map((city: any) => (
                  <option key={city.id} value={city.id}>
                    {city.name}, {city.country}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">Scenario Name *</label>
              <input
                value={scenarioName}
                onChange={(e: any) => setScenarioName(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                placeholder="e.g., Green City 2030"
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">Description</label>
              <textarea
                value={description}
                onChange={(e: any) => setDescription(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                rows={3}
                placeholder="Describe your scenario..."
              />
            </div>

            <div>
              <label className="block text-sm font-medium mb-2 text-gray-300">Time Horizon *</label>
              <select
                value={timeHorizon}
                onChange={(e) => setTimeHorizon(e.target.value)}
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
              >
                <option value="1d">1 Day</option>
                <option value="7d">7 Days</option>
                <option value="30d">30 Days</option>
                <option value="1y">1 Year</option>
                <option value="5y">5 Years</option>
                <option value="10y">10 Years</option>
              </select>
            </div>

            <div className="border-t border-gray-700 pt-4">
              <h4 className="text-sm font-semibold text-white mb-3">Parameters</h4>
              
              <div className="mb-3">
                <label className="block text-sm text-gray-400 mb-2">Vegetation Change (%)</label>
                <input
                  type="number"
                  value={vegetationChange}
                  onChange={(e: any) => setVegetationChange(e.target.value)}
                  step="0.1"
                  min="-100"
                  max="100"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                />
              </div>

              <div>
                <label className="block text-sm text-gray-400 mb-2">Building Density Change (%)</label>
                <input
                  type="number"
                  value={buildingDensityChange}
                  onChange={(e: any) => setBuildingDensityChange(e.target.value)}
                  step="0.1"
                  min="-100"
                  max="100"
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white"
                />
              </div>
            </div>

            <button
              onClick={handleSubmit}
              disabled={!selectedCityId || createScenarioMutation.isPending}
              className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded text-white font-medium transition-colors"
            >
              {createScenarioMutation.isPending ? 'Creating...' : 'Create Scenario'}
            </button>

            <div className="border-t border-gray-700 pt-4">
              <h4 className="text-sm font-semibold text-white mb-3">Saved Scenarios ({scenarios?.length || 0})</h4>
              
              {scenarios && scenarios.length > 0 ? (
                <div className="space-y-2 max-h-64 overflow-y-auto">
                  {scenarios.map((scenario: any) => (
                    <div key={scenario.id} className="bg-gray-700 rounded p-3 border border-gray-600">
                      <div className="flex justify-between items-start mb-1">
                        <div className="font-semibold text-white text-sm">{scenario.name}</div>
                        <button
                          onClick={() => handleDeleteScenario(scenario.id)}
                          className="text-red-400 hover:text-red-300 text-xs"
                        >
                          🗑️
                        </button>
                      </div>
                      <div className="text-xs text-gray-400">{scenario.city_name || 'Unknown'}</div>
                      <div className="text-xs text-gray-500 mt-1">{scenario.time_horizon}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div className="text-center py-4 text-gray-400 text-sm">No scenarios yet</div>
              )}
            </div>
          </div>
        </div>
      </div>

      {!isPanelOpen && (
        <button
          onClick={() => setIsPanelOpen(true)}
          className="fixed top-4 right-4 px-4 py-2 bg-gray-800 hover:bg-gray-700 border border-gray-700 rounded-lg text-white shadow-lg transition-colors"
        >
          🎯 Open Panel
        </button>
      )}
    </div>
  )
}

