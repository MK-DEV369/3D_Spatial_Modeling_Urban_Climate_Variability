import { useState, useEffect, useCallback } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { osmApi, LayerType, BBox, GeoJSONCollection } from '../services/osm'

/**
 * Hook to fetch OSM features by bounding box
 */
export function useOSMByBBox(
  layerType: LayerType,
  
  bbox: BBox | null,
  options?: {
    scenario?: string
    active?: boolean
    enabled?: boolean
  }
) {
  const bboxKey = bbox ? `${bbox.minLon},${bbox.minLat},${bbox.maxLon},${bbox.maxLat}` : null;
  return useQuery<GeoJSONCollection>({
    
    queryKey: ['osm', layerType, 'bbox', bboxKey, options?.scenario, options?.active],
    queryFn: () => {
      if (!bbox) throw new Error('BBox is required')
      return osmApi.getByBBox(layerType, bbox, options)
    },
    enabled: !!bbox && (options?.enabled !== false),
    staleTime: 60000,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,
  })
}

/**
 * Hook to fetch all OSM features
 */
export function useOSMAll(
  layerType: LayerType,
  options?: {
    scenario?: string
    active?: boolean
    enabled?: boolean
  }
) {
  return useQuery<GeoJSONCollection>({
    queryKey: ['osm', layerType, 'all', options?.scenario, options?.active],
    queryFn: () => osmApi.getAll(layerType, options),
    enabled: options?.enabled !== false,
    refetchOnWindowFocus: false,
    refetchOnReconnect: false,

  })
}

/**
 * Hook to fetch single OSM feature by ID
 */
export function useOSMById(layerType: LayerType, osmId: number | null) {
  return useQuery({
    queryKey: ['osm', layerType, osmId],
    queryFn: () => {
      if (!osmId) throw new Error('OSM ID is required')
      return osmApi.getById(layerType, osmId)
    },
    enabled: !!osmId,
  })
}

/**
 * Hook for OSM CRUD mutations
 */
export function useOSMMutations(layerType: LayerType) {
  const queryClient = useQueryClient()

  const createMutation = useMutation({
    mutationFn: (data: Parameters<typeof osmApi.create>[1]) =>
      osmApi.create(layerType, data),
    onSuccess: () => {
      // Invalidate all queries for this layer type
      queryClient.invalidateQueries({ queryKey: ['osm', layerType] })
    },
  })

  const updateMutation = useMutation({
    mutationFn: ({ osmId, data }: { osmId: number; data: Parameters<typeof osmApi.update>[2] }) =>
      osmApi.update(layerType, osmId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['osm', layerType] })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: (osmId: number) => osmApi.delete(layerType, osmId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['osm', layerType] })
    },
  })

  return {
    create: createMutation,
    update: updateMutation,
    delete: deleteMutation,
  }
}
