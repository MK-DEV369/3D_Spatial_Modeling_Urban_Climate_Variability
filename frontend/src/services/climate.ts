import api from './api'

export const climateApi = {
  getPredictions: async (scenarioId: number): Promise<any[]> => {
    const response = await api.get(`/scenarios/${scenarioId}/predictions/`)
    return response.data
  },

  generatePrediction: async (data: {
    city_id: number
    time_horizon: string
    parameters?: Record<string, any>
  }): Promise<any> => {
    const response = await api.post('/predictions/climate/', data)
    return response.data
  },
}

