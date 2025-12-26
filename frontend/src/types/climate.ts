export interface ClimateData {
  id: number
  city: number
  city_name: string
  timestamp: string
  temperature: number
  humidity?: number
  precipitation?: number
  wind_speed?: number
  wind_direction?: number
  pressure?: number
  solar_radiation?: number
  metadata?: Record<string, any>
}

export interface ClimatePrediction {
  timestamp: string
  temperature: number
  humidity?: number
  precipitation?: number
  [key: string]: any
}

