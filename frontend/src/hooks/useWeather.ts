import { useQuery } from '@tanstack/react-query'
import { weatherApi } from '../services/weather'

export function useWeather(lat: number, lon: number) {
  return useQuery({
    queryKey: ['weather', lat, lon],
    queryFn: () => weatherApi.getCurrent(lat, lon),
    enabled: lat !== 0 && lon !== 0,
    staleTime: 300000, // 5 minutes
  })
}

export function useWeatherForecast(lat: number, lon: number, days: number = 7) {
  return useQuery({
    queryKey: ['weather-forecast', lat, lon, days],
    queryFn: () => weatherApi.getForecast(lat, lon, days),
    enabled: lat !== 0 && lon !== 0,
    staleTime: 300000, // 5 minutes
  })
}
