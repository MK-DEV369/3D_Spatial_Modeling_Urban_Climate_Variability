import api from './api'
import { City, BuildingGeoJSON } from '../types/city'

export const citiesApi = {
  getAll: async (): Promise<City[]> => {
    const response = await api.get('/cities/')
    return response.data
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

