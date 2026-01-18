# MapViewer Components Comparison

## Overview

This document explains the differences between `MapViewer.tsx` and `MapViewerGL.tsx`, and how they've been updated to use backend OSM buildings.

## Component Differences

### MapViewer.tsx (2D Leaflet)
- **Library**: React-Leaflet (Leaflet.js wrapper)
- **Rendering**: 2D canvas-based rendering
- **Performance**: Good for simple 2D maps, less performant with many features
- **Features**:
  - 2D building polygons
  - Heat island visualization (circles)
  - Multiple map styles (street, satellite, topo)
  - Building popups with details
  - Measurement tools
- **Data Source**: Previously fetched from Overpass API directly
- **Use Case**: Simple 2D visualization, lightweight maps

### MapViewerGL.tsx (3D GPU-Accelerated)
- **Library**: React-Map-GL (MapLibre GL JS wrapper)
- **Rendering**: WebGL/GPU-accelerated 3D rendering
- **Performance**: Excellent for complex 3D visualizations, handles thousands of buildings smoothly
- **Features**:
  - **3D building extrusion** (buildings rendered as 3D blocks)
  - Heat island visualization
  - Multiple map styles (street, dark, satellite)
  - Tilt/pitch controls for 3D viewing
  - Dynamic height-based coloring
- **Data Source**: **Now uses backend API** (`/api/buildings/bbox/`)
- **Use Case**: Advanced 3D visualization, urban planning, climate analysis

## Key Updates to MapViewerGL.tsx

### ✅ Backend Integration
- **Removed**: Direct Overpass API calls
- **Added**: Integration with backend API using `useOSMByBBox` hook
- **Benefits**:
  - Uses your PostgreSQL/PostGIS database
  - Supports scenario-based filtering
  - Respects `active` flag for layer toggling
  - Faster queries (local database vs external API)

### ✅ Building Height Calculation
Since the backend `BuildingsOSM` model doesn't include height data, the component now:
- Calculates height based on building type
- Uses sensible defaults:
  - Commercial: 20m
  - Office: 25m
  - Apartments: 18m
  - Residential/House: 8-10m
  - Industrial: 15m
  - Default: 10m

### ✅ Dynamic Bbox Updates
- Automatically fetches buildings when map is panned/zoomed
- Uses React Query for efficient caching and refetching
- Debounced updates to prevent excessive API calls

### ✅ Scenario Support
- Added scenario filter input
- Can filter buildings by `scenario_id` (e.g., "baseline", "future", etc.)
- Useful for comparing different urban planning scenarios

## Usage

### MapViewer.tsx
```tsx
<MapViewer 
  cityId={1}
  cityName="Bengaluru"
  latitude={12.9716}
  longitude={77.5946}
/>
```

### MapViewerGL.tsx
```tsx
<MapViewerGL 
  cityId={1}
  cityName="Bengaluru"
  latitude={12.9716}
  longitude={77.5946}
/>
```

## When to Use Which?

### Use MapViewer.tsx when:
- You need simple 2D visualization
- You want lightweight rendering
- You don't need 3D effects
- You're building a simple dashboard

### Use MapViewerGL.tsx when:
- You need 3D building visualization
- You want GPU-accelerated performance
- You're doing urban planning/climate analysis
- You need advanced visualization features
- You want to showcase the backend OSM data integration

## Backend API Endpoints Used

### MapViewerGL.tsx uses:
- `GET /api/buildings/bbox/?bbox=minLon,minLat,maxLon,maxLat&scenario=baseline&active=true`
  - Returns GeoJSON FeatureCollection
  - Filters by bounding box, scenario, and active status
  - Uses PostGIS spatial queries for performance

## Future Enhancements

1. **Add height field to BuildingsOSM model** - Store actual building heights from OSM
2. **Add building:levels support** - Parse and store floor count
3. **Add building:material** - For better visualization
4. **Add building:roof:shape** - For more realistic 3D rendering
5. **Add building:color** - For visual variety
