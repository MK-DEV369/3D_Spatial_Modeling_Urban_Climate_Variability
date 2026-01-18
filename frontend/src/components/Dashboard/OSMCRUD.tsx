import { useState, useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { useOSMMutations, useOSMById } from '../../hooks/useOSM'
import api from '../../services/api'

type LayerType = 'buildings' | 'roads' | 'water' | 'green'

interface OSMCRUDProps {
  layerType: LayerType
  onFeatureSelect?: (feature: any) => void
  onFeatureHighlight?: (osmId: number | null) => void
}

export default function OSMCRUD({ layerType, onFeatureSelect, onFeatureHighlight }: OSMCRUDProps) {
  const [osmId, setOsmId] = useState<string>('')
  const [name, setName] = useState<string>('')
  const [scenario, setScenario] = useState<string>('baseline')
  const [active, setActive] = useState<boolean>(true)
  const [message, setMessage] = useState<string>('')
  const [selectedOsmId, setSelectedOsmId] = useState<number | null>(null)
  
  const queryClient = useQueryClient()
  const mutations = useOSMMutations(layerType)
  
  // Fetch feature by ID when OSM ID is entered
  const { data: featureData } = useOSMById(
    layerType,
    selectedOsmId
  )

  // Notify parent when feature is loaded
  useEffect(() => {
    if (featureData && onFeatureSelect) {
      onFeatureSelect(featureData)
    }
    if (selectedOsmId !== null && onFeatureHighlight) {
      onFeatureHighlight(selectedOsmId)
    }
  }, [featureData, selectedOsmId, onFeatureSelect, onFeatureHighlight])

  const layerLabels = {
    buildings: 'Buildings',
    roads: 'Roads',
    water: 'Water Bodies',
    green: 'Green Spaces'
  }

  const handleCreate = async () => {
    setMessage('')
    try {
      const response = await mutations.create.mutateAsync({
        osm_id: parseInt(osmId) || Date.now(),
        name: name || undefined,
        active,
        scenario_id: scenario,
        geom: {
          type: 'Polygon',
          coordinates: [[[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]]
        }
      })
      setMessage(`✅ Created ${layerLabels[layerType]} successfully!`)
      const createdOsmId = response?.properties?.osm_id || parseInt(osmId) || Date.now()
      setOsmId('')
      setName('')
      // Invalidate queries to refresh map
      queryClient.invalidateQueries({ queryKey: ['osm', layerType] })
      // Highlight the newly created feature
      setSelectedOsmId(createdOsmId)
    } catch (error: any) {
      setMessage(`❌ Error: ${error.response?.data?.error || error.message}`)
    }
  }

  const handleRead = async () => {
    if (!osmId) {
      setMessage('❌ Please enter OSM ID')
      return
    }
    setMessage('')
    try {
      const id = parseInt(osmId)
      setSelectedOsmId(id) // This will trigger useOSMById hook
      const response = await api.get(`/${layerType}/${id}/`)
      const props = response.data.properties || response.data
      setName(props.name || '')
      setActive(props.active ?? true)
      setScenario(props.scenario_id || 'baseline')
      setMessage(`✅ Found ${layerLabels[layerType]}! Highlighting on map...`)
    } catch (error: any) {
      setMessage(`❌ Error: ${error.response?.data?.detail || 'Not found'}`)
      setSelectedOsmId(null)
    }
  }
  
  const handleClearHighlight = () => {
    setSelectedOsmId(null)
    if (onFeatureHighlight) {
      onFeatureHighlight(null)
    }
    setMessage('')
  }

  const handleUpdate = async () => {
    if (!osmId) {
      setMessage('❌ Please enter OSM ID')
      return
    }
    setMessage('')
    try {
      await mutations.update.mutateAsync({
        osmId: parseInt(osmId),
        data: {
          name: name || undefined,
          active,
          scenario_id: scenario,
        }
      })
      setMessage(`✅ Updated ${layerLabels[layerType]} successfully!`)
      // Invalidate queries to refresh map
      queryClient.invalidateQueries({ queryKey: ['osm', layerType] })
    } catch (error: any) {
      setMessage(`❌ Error: ${error.response?.data?.error || error.message}`)
    }
  }

  const handleDelete = async () => {
    if (!osmId) {
      setMessage('❌ Please enter OSM ID')
      return
    }
    if (!confirm(`Are you sure you want to delete ${layerLabels[layerType]} ${osmId}?`)) {
      return
    }
    setMessage('')
    try {
      await mutations.delete.mutateAsync(parseInt(osmId))
      setMessage(`✅ Deleted ${layerLabels[layerType]} successfully!`)
      setOsmId('')
      setName('')
      // Invalidate queries to refresh map
      queryClient.invalidateQueries({ queryKey: ['osm', layerType] })
    } catch (error: any) {
      setMessage(`❌ Error: ${error.response?.data?.error || error.message}`)
    }
  }

  return (
    <div className="space-y-4">
      <h3 className="text-lg font-semibold text-white mb-3">
        {layerLabels[layerType]} CRUD Operations
      </h3>

      {message && (
        <div className={`p-3 rounded text-sm ${
          message.includes('✅') 
            ? 'bg-green-900/30 border border-green-700 text-green-200' 
            : 'bg-red-900/30 border border-red-700 text-red-200'
        }`}>
          {message}
        </div>
      )}

      <div className="space-y-3">
        <div>
          <label className="block text-sm text-gray-400 mb-2">OSM ID</label>
          <input
            type="text"
            value={osmId}
            onChange={(e) => setOsmId(e.target.value)}
            placeholder="Enter OSM ID"
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-500"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">Name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Enter name"
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-500"
          />
        </div>

        <div>
          <label className="block text-sm text-gray-400 mb-2">Scenario</label>
          <input
            type="text"
            value={scenario}
            onChange={(e) => setScenario(e.target.value)}
            placeholder="baseline"
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white placeholder-gray-500"
          />
        </div>

        <div className="flex items-center space-x-2">
          <input
            type="checkbox"
            id={`active-${layerType}`}
            checked={active}
            onChange={(e) => setActive(e.target.checked)}
            className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
          />
          <label htmlFor={`active-${layerType}`} className="text-sm text-gray-300">
            Active
          </label>
        </div>

        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={handleCreate}
            disabled={mutations.create.isPending}
            className="px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded text-white font-medium transition-colors"
          >
            {mutations.create.isPending ? 'Creating...' : 'Create'}
          </button>
          <button
            onClick={handleRead}
            disabled={!osmId}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-white font-medium transition-colors"
          >
            Read & Highlight
          </button>
          <button
            onClick={handleUpdate}
            disabled={mutations.update.isPending || !osmId}
            className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 disabled:bg-gray-600 rounded text-white font-medium transition-colors"
          >
            {mutations.update.isPending ? 'Updating...' : 'Update'}
          </button>
          <button
            onClick={handleDelete}
            disabled={mutations.delete.isPending || !osmId}
            className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 rounded text-white font-medium transition-colors"
          >
            {mutations.delete.isPending ? 'Deleting...' : 'Delete'}
          </button>
        </div>
        
        {selectedOsmId && (
          <button
            onClick={handleClearHighlight}
            className="w-full mt-2 px-4 py-2 bg-gray-600 hover:bg-gray-700 rounded text-white font-medium transition-colors"
          >
            Clear Highlight
          </button>
        )}
      </div>

      <div className="bg-gray-700/50 rounded p-3 text-sm text-gray-300">
        <div className="font-semibold mb-2">Operations:</div>
        <div className="space-y-1">
          <div>• Create: Add new {layerLabels[layerType].toLowerCase()}</div>
          <div>• Read: Fetch by OSM ID</div>
          <div>• Update: Modify existing record</div>
          <div>• Delete: Remove record</div>
        </div>
      </div>
    </div>
  )
}
