import api from './api'

export type LayerType = 'buildings' | 'roads' | 'water' | 'green'

export interface BBox {
  minLon: number
  minLat: number
  maxLon: number
  maxLat: number
}

export interface OSMFeature {
  type: 'Feature'
  geometry: {
    type: string
    coordinates: any
  }
  properties: {
    osm_id: number
    name?: string
    active: boolean
    scenario_id: string
    modified_at: string
    [key: string]: any // Additional properties vary by layer type
  }
}

export interface GeoJSONCollection {
  type: 'FeatureCollection'
  features: OSMFeature[]
}

export const osmApi = {
  /**
   * Get OSM features within a bounding box
   */
  getByBBox: async (
    layerType: LayerType,
    bbox: BBox,
    options?: {
      scenario?: string
      active?: boolean
    }
  ): Promise<GeoJSONCollection> => {
    const params = new URLSearchParams({
      bbox: `${bbox.minLon},${bbox.minLat},${bbox.maxLon},${bbox.maxLat}`,
    })
    
    if (options?.scenario) {
      params.append('scenario', options.scenario)
    }
    
    if (options?.active !== undefined) {
      params.append('active', options.active.toString())
    }

    const response = await api.get(`/${layerType}/bbox/?${params.toString()}`)
    return response.data
  },

  /**
   * Get all OSM features (paginated)
   */
  getAll: async (
    layerType: LayerType,
    options?: {
      scenario?: string
      active?: boolean
    }
  ): Promise<GeoJSONCollection> => {
    const params = new URLSearchParams()
    
    if (options?.scenario) {
      params.append('scenario', options.scenario)
    }
    
    if (options?.active !== undefined) {
      params.append('active', options.active.toString())
    }

    const queryString = params.toString()
    const url = `/${layerType}/${queryString ? `?${queryString}` : ''}`
    const response = await api.get(url)
    
    // Handle DRF paginated response or direct FeatureCollection
    if (response.data.type === 'FeatureCollection') {
      return response.data
    }
    
    // If paginated, extract features
    if (response.data.results) {
      return {
        type: 'FeatureCollection',
        features: response.data.results.map((item: any) => ({
          type: 'Feature',
          geometry: item.geometry,
          properties: item.properties || item,
        })),
      }
    }
    
    return response.data
  },

  /**
   * Get single OSM feature by ID
   */
  getById: async (layerType: LayerType, osmId: number): Promise<OSMFeature> => {
    const response = await api.get(`/${layerType}/${osmId}/`)
    return response.data
  },

  /**
   * Create new OSM feature
   */
  create: async (layerType: LayerType, data: Partial<OSMFeature['properties']> & { geom: any }): Promise<OSMFeature> => {
    const response = await api.post(`/${layerType}/`, {
      ...data,
      modified_at: new Date().toISOString(),
    })
    return response.data
  },

  /**
   * Update OSM feature
   */
  update: async (
    layerType: LayerType,
    osmId: number,
    data: Partial<OSMFeature['properties']>
  ): Promise<OSMFeature> => {
    const response = await api.patch(`/${layerType}/${osmId}/`, {
      ...data,
      modified_at: new Date().toISOString(),
    })
    return response.data
  },

  /**
   * Delete OSM feature
   */
  delete: async (layerType: LayerType, osmId: number): Promise<void> => {
    await api.delete(`/${layerType}/${osmId}/`)
  },
}
