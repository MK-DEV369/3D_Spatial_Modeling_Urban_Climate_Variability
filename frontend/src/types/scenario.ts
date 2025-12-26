export interface Scenario {
  id: number
  name: string
  description?: string
  city: number
  city_name: string
  parameters: Record<string, any>
  time_horizon: '1d' | '7d' | '30d' | '1y' | '5y' | '10y'
  created_at: string
  updated_at: string
}

export interface Prediction {
  id: number
  scenario: number
  scenario_name: string
  model_type: 'weather' | 'climate' | 'traffic' | 'pollution'
  timestamp: string
  predictions: Record<string, any>
  created_at: string
}

