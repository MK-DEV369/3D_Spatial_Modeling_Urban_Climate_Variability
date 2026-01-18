import { useEffect, useState, useRef } from 'react';
import { MapContainer, TileLayer, useMap, GeoJSON, Marker, Popup, Circle, LayersControl } from 'react-leaflet';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import './MapViewer.css';
import { Button } from '@/components/ui/button';
import { Card } from '@/components/ui/card';
import { FadeIn } from '../reactbits';

// Fix Leaflet default icon issue with Vite
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png',
  iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png',
  shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png',
});

interface MapViewerProps {
  cityId?: number;
  cityName?: string;
  latitude?: number;
  longitude?: number;
}

interface BuildingData {
  type: string;
  geometry: any;
  properties: {
    height?: number;
    levels?: number;
    name?: string;
    building?: string;
  };
}

function MapController({ center, zoom }: { center: [number, number]; zoom: number }) {
  const map = useMap();
  useEffect(() => {
    map.setView(center, zoom);
  }, [center, zoom, map]);
  return null;
}

export default function MapViewer({ cityId, cityName, latitude = 12.9716, longitude = 77.5946 }: MapViewerProps) {
  const [buildings, setBuildings] = useState<BuildingData[]>([]);
  const [loadingBuildings, setLoadingBuildings] = useState(false);
  const [showBuildings, setShowBuildings] = useState(true);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [measuring, setMeasuring] = useState(false);
  const [mapStyle, setMapStyle] = useState<'street' | 'satellite' | 'topo'>('street');
  const [selectedBuilding, setSelectedBuilding] = useState<BuildingData | null>(null);
  const mapRef = useRef<any>(null);

  // Fetch buildings from OpenStreetMap Overpass API
  const fetchBuildings = async (lat: number, lon: number, radius: number = 0.01) => {
    setLoadingBuildings(true);
    try {
      const bbox = `${lat - radius},${lon - radius},${lat + radius},${lon + radius}`;
      const overpassUrl = `https://overpass-api.de/api/interpreter?data=[out:json];(way["building"](${bbox});relation["building"](${bbox}););out geom;`;
      
      const response = await fetch(overpassUrl);
      const data = await response.json();
      
      // Convert OSM data to GeoJSON
      const geojsonBuildings = data.elements
        .filter((el: any) => el.geometry || el.members)
        .map((el: any) => ({
          type: 'Feature',
          geometry: {
            type: el.type === 'way' ? 'Polygon' : 'MultiPolygon',
            coordinates: el.geometry ? [[el.geometry.map((pt: any) => [pt.lon, pt.lat])]] : []
          },
          properties: {
            height: el.tags?.['height'] || el.tags?.['building:levels'] ? parseInt(el.tags?.['building:levels']) * 3 : 10,
            levels: el.tags?.['building:levels'] || 3,
            name: el.tags?.name || 'Building',
            building: el.tags?.building || 'yes'
          }
        }));
      
      setBuildings(geojsonBuildings);
    } catch (error) {
      console.error('Error fetching buildings:', error);
    } finally {
      setLoadingBuildings(false);
    }
  };

  useEffect(() => {
    if (latitude && longitude) {
      fetchBuildings(latitude, longitude);
    }
  }, [latitude, longitude]);

  const buildingStyle = (feature: any) => {
    const height = feature?.properties?.height || 10;
    const opacity = Math.min(height / 50, 0.8);
    return {
      fillColor: height > 30 ? '#ef4444' : height > 15 ? '#f59e0b' : '#3b82f6',
      weight: 1,
      opacity: 1,
      color: 'white',
      fillOpacity: opacity
    };
  };

  const onEachBuilding = (feature: any, layer: any) => {
    if (feature.properties) {
      const { name, height, levels, building } = feature.properties;
      layer.bindPopup(`
        <div class="p-2">
          <h3 class="font-bold text-lg">${name}</h3>
          <p class="text-sm">Type: ${building}</p>
          <p class="text-sm">Height: ~${height}m</p>
          <p class="text-sm">Floors: ${levels}</p>
        </div>
      `);
      layer.on('click', () => setSelectedBuilding(feature));
    }
  };

  const tileUrls = {
    street: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    satellite: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    topo: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png'
  };

  const heatmapPoints = [
    { position: [latitude + 0.005, longitude + 0.005] as [number, number], intensity: 80 },
    { position: [latitude - 0.003, longitude + 0.003] as [number, number], intensity: 60 },
    { position: [latitude + 0.002, longitude - 0.004] as [number, number], intensity: 90 }
  ];

  return (
    <Card className="overflow-hidden relative">
      <FadeIn duration={800}>
        <div className="h-[600px] relative">
          {/* Map Controls Toolbar */}
          <div className="absolute top-4 right-4 z-[1000] space-y-2">
            <Card className="p-2 space-y-2 bg-white/95 backdrop-blur">
              <div className="text-xs font-semibold text-gray-700">Map Style</div>
              <div className="flex gap-1">
                <Button
                  size="sm"
                  variant={mapStyle === 'street' ? 'default' : 'outline'}
                  onClick={() => setMapStyle('street')}
                  className="text-xs"
                >
                  🗺️ Street
                </Button>
                <Button
                  size="sm"
                  variant={mapStyle === 'satellite' ? 'default' : 'outline'}
                  onClick={() => setMapStyle('satellite')}
                  className="text-xs"
                >
                  🛰️ Satellite
                </Button>
                <Button
                  size="sm"
                  variant={mapStyle === 'topo' ? 'default' : 'outline'}
                  onClick={() => setMapStyle('topo')}
                  className="text-xs"
                >
                  🏔️ Topo
                </Button>
              </div>

              <div className="text-xs font-semibold text-gray-700 pt-2">Layers</div>
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-xs cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showBuildings}
                    onChange={(e) => setShowBuildings(e.target.checked)}
                    className="rounded"
                  />
                  🏢 Buildings ({buildings.length})
                </label>
                <label className="flex items-center gap-2 text-xs cursor-pointer">
                  <input
                    type="checkbox"
                    checked={showHeatmap}
                    onChange={(e) => setShowHeatmap(e.target.checked)}
                    className="rounded"
                  />
                  🔥 Heat Islands
                </label>
              </div>

              <div className="text-xs font-semibold text-gray-700 pt-2">Tools</div>
              <div className="space-y-1">
                <Button
                  size="sm"
                  variant={measuring ? 'default' : 'outline'}
                  onClick={() => setMeasuring(!measuring)}
                  className="w-full text-xs"
                >
                  📏 Measure Distance
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => fetchBuildings(latitude, longitude, 0.02)}
                  disabled={loadingBuildings}
                  className="w-full text-xs"
                >
                  {loadingBuildings ? '⏳ Loading...' : '🔄 Reload Buildings'}
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => window.open(`/viewer3d?lat=${latitude}&lon=${longitude}`, '_blank')}
                  className="w-full text-xs"
                >
                  🎨 View in 3D
                </Button>
              </div>
            </Card>
          </div>

          {/* Selected Building Info */}
          {selectedBuilding && (
            <div className="absolute bottom-4 left-4 z-[1000] max-w-xs">
              <Card className="p-3 bg-white/95 backdrop-blur">
                <div className="flex justify-between items-start mb-2">
                  <h3 className="font-bold text-sm">{selectedBuilding.properties.name}</h3>
                  <button
                    onClick={() => setSelectedBuilding(null)}
                    className="text-gray-500 hover:text-gray-700"
                  >
                    ✕
                  </button>
                </div>
                <div className="text-xs space-y-1 text-gray-600">
                  <p>📦 Type: {selectedBuilding.properties.building}</p>
                  <p>📏 Height: ~{selectedBuilding.properties.height}m</p>
                  <p>🏢 Floors: {selectedBuilding.properties.levels}</p>
                </div>
              </Card>
            </div>
          )}

          {/* Map Legend */}
          <div className="absolute bottom-4 right-4 z-[1000]">
            <Card className="p-2 bg-white/95 backdrop-blur">
              <div className="text-xs font-semibold text-gray-700 mb-1">Building Height</div>
              <div className="space-y-1 text-xs">
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-blue-500 rounded"></div>
                  <span>Low (&lt;15m)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-amber-500 rounded"></div>
                  <span>Medium (15-30m)</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-4 h-4 bg-red-500 rounded"></div>
                  <span>High (&gt;30m)</span>
                </div>
              </div>
            </Card>
          </div>

          <MapContainer
            center={[latitude, longitude]}
            zoom={15}
            style={{ height: '100%', width: '100%' }}
            ref={mapRef}
          >
            <MapController center={[latitude, longitude]} zoom={15} />
            
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url={tileUrls[mapStyle]}
            />

            {showBuildings && buildings.map((building, idx) => (
              <GeoJSON
                key={`building-${idx}`}
                data={building as any}
                style={buildingStyle}
                onEachFeature={onEachBuilding}
              />
            ))}

            {showHeatmap && heatmapPoints.map((point, idx) => (
              <Circle
                key={`heat-${idx}`}
                center={point.position}
                radius={500}
                pathOptions={{
                  fillColor: 'red',
                  fillOpacity: point.intensity / 200,
                  color: 'red',
                  weight: 1
                }}
              >
                <Popup>
                  <div className="text-xs">
                    <strong>Heat Island</strong><br />
                    Intensity: {point.intensity}%
                  </div>
                </Popup>
              </Circle>
            ))}

            <Marker position={[latitude, longitude]}>
              <Popup>
                <div className="text-center">
                  <strong>{cityName || 'City Center'}</strong>
                </div>
              </Popup>
            </Marker>
          </MapContainer>
        </div>
      </FadeIn>
    </Card>
  );
}
