import { useQuery } from '@tanstack/react-query'
import { citiesApi } from '../services/cities'
import { City, BuildingGeoJSON } from '../types/city'

export const useCities = () => {
  return useQuery<City[]>({
    queryKey: ['cities'],
    queryFn: () => citiesApi.getAll(),
  })
}

export const useCity = (id: number) => {
  return useQuery<City>({
    queryKey: ['cities', id],
    queryFn: () => citiesApi.getById(id),
    enabled: !!id,
  })
}

export const useCityBuildings = (cityId: number) => {
  return useQuery<BuildingGeoJSON>({
    queryKey: ['cities', cityId, 'buildings'],
    queryFn: () => citiesApi.getBuildings(cityId),
    enabled: !!cityId,
  })
}

export const useCityClimate = (cityId: number) => {
  return useQuery({
    queryKey: ['cities', cityId, 'climate'],
    queryFn: () => citiesApi.getClimate(cityId),
    enabled: !!cityId,
  })
}

export const useCityTraffic = (cityId: number) => {
  return useQuery({
    queryKey: ['cities', cityId, 'traffic'],
    queryFn: () => citiesApi.getTraffic(cityId),
    enabled: !!cityId,
  })
}

export const useCityPollution = (cityId: number) => {
  return useQuery({
    queryKey: ['cities', cityId, 'pollution'],
    queryFn: () => citiesApi.getPollution(cityId),
    enabled: !!cityId,
  })
}

