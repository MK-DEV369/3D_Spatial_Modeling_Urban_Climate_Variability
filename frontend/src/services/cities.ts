import api from './api'
import { City, BuildingGeoJSON } from '../types/city'

export const citiesApi = {
  getAll: async (): Promise<City[]> => {
    const response = await api.get('/cities/')
    const data = response.data
    // Support both paginated and non-paginated DRF responses
    if (Array.isArray(data)) return data
    if (Array.isArray(data?.results)) return data.results
    // Fallback: if neither, return empty array to avoid runtime errors
    console.warn('Unexpected /cities/ response shape; returning empty list')
    return []
  },

  getById: async (id: number): Promise<City> => {
    const response = await api.get(`/cities/${id}/`)
    return response.data
  },

  getBuildings: async (id: number): Promise<BuildingGeoJSON> => {
    const response = await api.get(`/cities/${id}/buildings/`)
    return response.data
  },

  getClimate: async (id: number): Promise<any[]> => {
    const response = await api.get(`/cities/${id}/climate/`)
    return response.data
  },

  getTraffic: async (id: number): Promise<any[]> => {
    const response = await api.get(`/cities/${id}/traffic/`)
    return response.data
  },

  getPollution: async (id: number): Promise<any[]> => {
    const response = await api.get(`/cities/${id}/pollution/`)
    return response.data
  },
}

