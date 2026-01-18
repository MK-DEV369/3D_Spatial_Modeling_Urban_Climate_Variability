import { useEffect, useState, useRef } from 'react'
import Map, { Source, Layer, NavigationControl, ScaleControl } from 'react-map-gl/maplibre'
import type { MapRef, ViewStateChangeEvent } from 'react-map-gl/maplibre'
import 'maplibre-gl/dist/maplibre-gl.css'
import { Button } from '../ui/button'
import { Card } from '../ui/card'
import { FadeIn } from '../reactbits'
import { useOSMByBBox } from '../../hooks/useOSM'

interface MapViewerProps {
    cityId: number
    cityName?: string
    latitude: number
    longitude: number
}

const heatIslandData = {
    type: 'FeatureCollection' as const,
    features: [
        {
            type: 'Feature' as const,
            geometry: { type: 'Point' as const, coordinates: [77.5946, 12.9716] },
            properties: { intensity: 0.8 }
        },
        {
            type: 'Feature' as const,
            geometry: { type: 'Point' as const, coordinates: [77.5960, 12.9730] },
            properties: { intensity: 0.6 }
        },
        {
            type: 'Feature' as const,
            geometry: { type: 'Point' as const, coordinates: [77.5930, 12.9700] },
            properties: { intensity: 0.7 }
        }
    ]
}

export default function MapViewer({ cityId, cityName, latitude, longitude }: MapViewerProps) {
    const [mapStyle, setMapStyle] = useState<'street' | 'satellite' | 'dark'>('dark')
    const [showBuildings, setShowBuildings] = useState(true)
    const [show3DBuildings, setShow3DBuildings] = useState(true)
    const [showHeatIslands, setShowHeatIslands] = useState(true)
    const [scenario, setScenario] = useState<string>('baseline')
    const [mapBounds, setMapBounds] = useState<{ minLon: number; minLat: number; maxLon: number; maxLat: number } | null>(null)
    const mapRef = useRef<MapRef>(null)

    const [viewState, setViewState] = useState({
        longitude,
        latitude,
        zoom: 14,
        pitch: 45,
        bearing: 0
    })

    // Fetch buildings from backend API
    const { data: buildingsData, isLoading: loading } = useOSMByBBox(
        'buildings',
        mapBounds,
        { scenario, active: true, enabled: showBuildings && !!mapBounds }
    )

    useEffect(() => {
        setViewState(prev => ({
            ...prev,
            longitude,
            latitude
        }))
        // Set initial bounds around city
        const offset = 0.05 // ~5km
        setMapBounds({
            minLon: longitude - offset,
            minLat: latitude - offset,
            maxLon: longitude + offset,
            maxLat: latitude + offset,
        })
    }, [cityId, latitude, longitude])

    // Update bounds when map moves
    const handleMoveEnd = () => {
        const map = mapRef.current?.getMap()
        if (map) {
            const bounds = map.getBounds()
            setMapBounds({
                minLon: bounds.getWest(),
                minLat: bounds.getSouth(),
                maxLon: bounds.getEast(),
                maxLat: bounds.getNorth(),
            })
        }
    }

    // Process buildings data for 3D extrusion
    const processedBuildings = buildingsData ? {
        ...buildingsData,
        features: buildingsData.features.map((feature: any) => {
            const props = feature.properties || {}
            const buildingType = props.building || 'yes'
            
            // Calculate height based on building type (defaults)
            // Commercial/industrial buildings are taller, residential shorter
            const heightDefaults: Record<string, number> = {
                'commercial': 20,
                'industrial': 15,
                'retail': 12,
                'office': 25,
                'apartments': 18,
                'residential': 10,
                'house': 8,
                'yes': 10, // default for unclassified buildings
            }
            
            const height = heightDefaults[buildingType.toLowerCase()] || 10

            return {
                ...feature,
                properties: {
                    ...props,
                    height,
                    name: props.name || undefined,
                    building_type: buildingType,
                }
            }
        })
    } : null

    const getMapStyleUrl = () => {
        switch (mapStyle) {
            case 'satellite':
                return 'https://api.maptiler.com/maps/hybrid/style.json?key=get_your_own_OpIi9ZULNHzrESv6T2vL'
            case 'dark':
                return 'https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json'
            default:
                return 'https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
        }
    }

    const buildingLayer: any = {
        id: 'buildings-3d',
        type: 'fill-extrusion',
        source: 'buildings',
        paint: {
            'fill-extrusion-color': [
                'interpolate',
                ['linear'],
                ['get', 'height'],
                0, '#3b82f6',
                15, '#f59e0b',
                30, '#ef4444'
            ],
            'fill-extrusion-height': ['get', 'height'],
            'fill-extrusion-base': 0,
            'fill-extrusion-opacity': 0.8
        }
    }

    const heatIslandLayer: any = {
        id: 'heat-islands',
        type: 'circle',
        source: 'heat-islands',
        paint: {
            'circle-radius': [
                'interpolate',
                ['linear'],
                ['zoom'],
                12, 20,
                16, 80
            ],
            'circle-color': '#ef4444',
            'circle-opacity': ['*', ['get', 'intensity'], 0.3],
            'circle-stroke-color': '#dc2626',
            'circle-stroke-width': 2
        }
    }

    return (
        <FadeIn duration={800} delay={400}>
            <Card className="p-4 bg-gray-900/50 backdrop-blur-sm border-gray-800">
                <div className="flex justify-between items-center mb-4">
                    <h3 className="text-xl font-semibold">
                        {cityName || 'City'} - GPU-Accelerated 3D Map
                    </h3>
                    <div className="flex gap-2">
                        <Button
                            size="sm"
                            variant={mapStyle === 'street' ? 'default' : 'outline'}
                            onClick={() => setMapStyle('street')}
                        >
                            Street
                        </Button>
                        <Button
                            size="sm"
                            variant={mapStyle === 'dark' ? 'default' : 'outline'}
                            onClick={() => setMapStyle('dark')}
                        >
                            Dark
                        </Button>
                        <Button
                            size="sm"
                            variant={mapStyle === 'satellite' ? 'default' : 'outline'}
                            onClick={() => setMapStyle('satellite')}
                        >
                            Satellite
                        </Button>
                    </div>
                </div>

                <div className="mb-4 flex gap-4 flex-wrap">
                    <label className="flex items-center gap-2 text-sm">
                        <input
                            type="checkbox"
                            checked={showBuildings}
                            onChange={(e) => setShowBuildings(e.target.checked)}
                            className="rounded"
                        />
                        OSM Buildings
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                        <input
                            type="checkbox"
                            checked={show3DBuildings}
                            onChange={(e) => setShow3DBuildings(e.target.checked)}
                            className="rounded"
                        />
                        3D Extrusion
                    </label>
                    <label className="flex items-center gap-2 text-sm">
                        <input
                            type="checkbox"
                            checked={showHeatIslands}
                            onChange={(e) => setShowHeatIslands(e.target.checked)}
                            className="rounded"
                        />
                        Heat Islands
                    </label>
                    <div className="flex items-center gap-2 text-sm">
                        <label className="text-xs text-gray-400">Scenario:</label>
                        <input
                            type="text"
                            value={scenario}
                            onChange={(e) => setScenario(e.target.value)}
                            placeholder="baseline"
                            className="px-2 py-1 bg-gray-800 border border-gray-700 rounded text-sm text-white w-24"
                        />
                    </div>
                    {loading && (
                        <div className="text-sm text-yellow-400">Loading buildings...</div>
                    )}
                    <Button
                        size="sm"
                        variant="outline"
                        onClick={() => {
                            setViewState(prev => ({
                                ...prev,
                                pitch: prev.pitch === 0 ? 45 : 0,
                                bearing: 0
                            }))
                        }}
                    >
                        Toggle 3D View
                    </Button>
                    <a href="/viewer3d" className="text-blue-400 hover:text-blue-300 text-sm flex items-center">
                        Full 3D Viewer →
                    </a>
                </div>

                <div className="relative h-[600px] rounded-lg overflow-hidden">
                    <Map
                        ref={mapRef}
                        {...viewState}
                        onMove={(evt: ViewStateChangeEvent) => setViewState(evt.viewState)}
                        onMoveEnd={handleMoveEnd}
                        onLoad={handleMoveEnd}
                        mapStyle={getMapStyleUrl()}
                        style={{ width: '100%', height: '100%' }}
                        attributionControl={false}
                    >
                        <NavigationControl position="top-left" />
                        <ScaleControl position="bottom-left" />

                        {showBuildings && processedBuildings && show3DBuildings && (
                            <Source id="buildings" type="geojson" data={processedBuildings}>
                                <Layer {...buildingLayer} />
                            </Source>
                        )}
                        
                        {showBuildings && processedBuildings && !show3DBuildings && (
                            <Source id="buildings-2d" type="geojson" data={processedBuildings}>
                                <Layer
                                    id="buildings-2d-fill"
                                    type="fill"
                                    paint={{
                                        'fill-color': '#3b82f6',
                                        'fill-opacity': 0.6,
                                        'fill-outline-color': '#1e40af'
                                    }}
                                />
                            </Source>
                        )}

                        {showHeatIslands && (
                            <Source id="heat-islands" type="geojson" data={heatIslandData}>
                                <Layer {...heatIslandLayer} />
                            </Source>
                        )}
                    </Map>

                    <div className="absolute bottom-4 right-4 bg-gray-900/95 p-3 rounded-lg text-xs space-y-1">
                        <p className="text-green-400 font-semibold">⚡ GPU Accelerated</p>
                        <p className="text-gray-400">MapLibre GL JS + WebGL</p>
                        <p className="text-gray-400">Pitch: {viewState.pitch}° | Zoom: {viewState.zoom.toFixed(1)}</p>
                        {processedBuildings && (
                            <p className="text-gray-400">Buildings: {processedBuildings.features.length}</p>
                        )}
                    </div>
                </div>
            </Card>
        </FadeIn>
    )
}
