export interface City {
  id: number
  name: string
  country: string
  latitude: number
  longitude: number
  bounds?: any
  metadata?: Record<string, any>
  created_at: string
}

export interface Building {
  id: number
  osm_id: number
  city: number
  city_name: string
  building_type?: string
  height?: number
  metadata?: Record<string, any>
  geometry: {
    type: string
    coordinates: number[][][]
  }
}

export interface BuildingGeoJSON {
  type: 'FeatureCollection'
  features: Array<{
    type: 'Feature'
    geometry: {
      type: string
      coordinates: number[][][]
    }
    properties: Omit<Building, 'geometry'>
  }>
}

