import { useMemo } from 'react'
import * as THREE from 'three'
import { ClimateData } from '../../types/climate'

interface ClimateOverlayProps {
  climateData?: ClimateData[]
  overlayType?: 'temperature' | 'humidity' | 'precipitation'
}

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

// Color mapping for precipitation (blue to dark blue)
const getPrecipitationColor = (precipitation: number): string => {
  const normalized = Math.max(0, Math.min(1, precipitation / 100))
  const r = Math.floor(0)
  const g = Math.floor(100 + normalized * 100)
  const b = Math.floor(200 + normalized * 55)
  return `rgb(${r}, ${g}, ${b})`
}

export function getClimateColor(
  climateData: ClimateData | undefined,
  overlayType: 'temperature' | 'humidity' | 'precipitation' = 'temperature'
): string {
  if (!climateData) {
    return '#8B9DC3' // Default gray-blue
  }

  switch (overlayType) {
    case 'temperature':
      return getTemperatureColor(climateData.temperature)
    case 'humidity':
      return getHumidityColor(climateData.humidity || 0)
    case 'precipitation':
      return getPrecipitationColor(climateData.precipitation || 0)
    default:
      return '#8B9DC3'
  }
}

export default function ClimateOverlay({ climateData, overlayType = 'temperature' }: ClimateOverlayProps) {
  const latestClimate = useMemo(() => {
    if (!climateData || climateData.length === 0) return undefined
    return climateData[0] // Most recent data point
  }, [climateData])

  // This component is mainly for utility functions
  // The actual color application happens in BuildingMesh
  return null
}

