import { ClimateData } from '../../types/climate'

// Color mapping function for temperature
const getTemperatureColor = (temp: number): string => {
  // Normalize temperature to 0-1 range (assuming -10°C to 50°C)
  const normalized = Math.max(0, Math.min(1, (temp + 10) / 60))
  
  // Blue (cold) to Red (hot) gradient
  if (normalized < 0.5) {
    // Blue to Cyan
    const t = normalized * 2
    const r = Math.floor(0)
    const g = Math.floor(100 + t * 155)
    const b = Math.floor(200 + t * 55)
    return `rgb(${r}, ${g}, ${b})`
  } else {
    // Cyan to Yellow to Red
    const t = (normalized - 0.5) * 2
    if (t < 0.5) {
      // Cyan to Yellow
      const t2 = t * 2
      const r = Math.floor(0 + t2 * 255)
      const g = Math.floor(255)
      const b = Math.floor(255 - t2 * 255)
      return `rgb(${r}, ${g}, ${b})`
    } else {
      // Yellow to Red
      const t2 = (t - 0.5) * 2
      const r = Math.floor(255)
      const g = Math.floor(255 - t2 * 255)
      const b = Math.floor(0)
      return `rgb(${r}, ${g}, ${b})`
    }
  }
}

// Color mapping for humidity (blue gradient)
const getHumidityColor = (humidity: number): string => {
  const normalized = Math.max(0, Math.min(1, humidity / 100))
  const r = Math.floor(50 + normalized * 50)
  const g = Math.floor(100 + normalized * 100)
  const b = Math.floor(200 + normalized * 55)
  return `rgb(${r}, ${g}, ${b})`
}

// Color mapping for precipitation (purple to blue gradient)
const getPrecipitationColor = (precip: number): string => {
  const normalized = Math.max(0, Math.min(1, precip / 50)) // Assuming 0-50mm range
  const r = Math.floor(200 - normalized * 100)
  const g = Math.floor(100)
  const b = Math.floor(255)
  return `rgb(${r}, ${g}, ${b})`
}

// Main function to get color based on overlay type
export const getClimateColor = (
  climate: ClimateData | undefined,
  overlayType: 'temperature' | 'humidity' | 'precipitation'
): string => {
  if (!climate) return '#808080' // Gray default

  switch (overlayType) {
    case 'temperature':
      return getTemperatureColor(climate.temperature)
    case 'humidity':
      return getHumidityColor(climate.humidity || 50)
    case 'precipitation':
      return getPrecipitationColor(climate.precipitation || 0)
    default:
      return '#808080'
  }
}

export default function ClimateOverlay() {
  return null // This is a utility component, no rendering
}
