# Feature Highlighting & Area Selection Guide

## Overview

The Dashboard now supports:
1. **Feature Highlighting by OSM ID** - Enter an OSM ID and highlight it on the map
2. **Area Selection** - Draw a rectangle to filter the map to a specific area
3. **Real-time CRUD Updates** - All Create, Update, Delete operations refresh the map automatically

## Features

### 1. Feature Highlighting by OSM ID

**How to use:**
1. Select a layer tab (Buildings, Roads, Water, or Green)
2. Enter an OSM ID in the "OSM ID" field
3. Click "Read & Highlight" button
4. The feature will be highlighted in **red** on the map
5. Click "Clear Highlight" to remove the highlight

**Visual Indicators:**
- Highlighted features appear in **red** with thicker borders
- Normal features appear in their default colors (orange for buildings, blue for roads, etc.)

### 2. Area Selection & Filtering

**How to use:**
1. Click "Select Area" button in the map controls (top-left)
2. Click on the map to set the first corner of the selection rectangle
3. Click again to set the opposite corner
4. The selected area will be shown with a green rectangle
5. Check "Filter by Area" to show only features within the selected area
6. Click "Clear Area" to remove the selection

**Benefits:**
- Focus on a specific region
- Reduce map clutter
- Faster rendering for large datasets
- Better performance when working with specific areas

### 3. CRUD Operations with Map Updates

All CRUD operations automatically update the map:

**Create:**
- Creates a new feature
- Automatically highlights the newly created feature
- Refreshes the map to show the new feature

**Read:**
- Fetches feature by OSM ID
- Highlights the feature on the map
- Populates the form with feature data

**Update:**
- Updates feature properties
- Refreshes the map to show changes
- Maintains highlight if the feature was highlighted

**Delete:**
- Removes the feature from the database
- Refreshes the map to remove the feature
- Clears the highlight

## Technical Implementation

### Component Structure

```
Dashboard.tsx
├── MapContainer (Leaflet)
│   ├── MapBoundsHandler (tracks map bounds)
│   ├── AreaSelector (handles rectangle selection)
│   ├── GeoJSON layers (buildings, roads, water, green)
│   └── Rectangle (shows selected area)
└── OSMCRUD (per layer type)
    ├── useOSMById (fetches feature by ID)
    ├── useOSMMutations (CRUD operations)
    └── Callbacks to Dashboard for highlighting
```

### State Management

- `highlightedOsmId`: Currently highlighted feature OSM ID
- `selectedArea`: Bounding box of selected area
- `filterByArea`: Whether to filter map by selected area
- `isSelectingArea`: Whether user is in area selection mode

### API Integration

- Uses `useOSMByBBox` hook for fetching features by bounding box
- Uses `useOSMById` hook for fetching single features
- Uses `useOSMMutations` hook for CRUD operations
- React Query handles caching and automatic refetching

## Usage Examples

### Example 1: Highlight a Building

1. Go to "Buildings" tab
2. Enter OSM ID: `123456`
3. Click "Read & Highlight"
4. Building with ID 123456 appears in red on the map

### Example 2: Filter by Area

1. Click "Select Area" button
2. Click at coordinates (12.97, 77.59) - first corner
3. Click at coordinates (12.98, 77.60) - second corner
4. Check "Filter by Area"
5. Map now shows only features within the selected rectangle

### Example 3: Create and Highlight

1. Go to "Buildings" tab
2. Enter OSM ID: `999999`
3. Enter Name: "New Building"
4. Click "Create"
5. New building is created and automatically highlighted on the map

## Performance Considerations

- **Bbox Queries**: Only fetches features within visible map bounds or selected area
- **Debounced Updates**: Map bounds updates are debounced (500ms) to prevent excessive API calls
- **React Query Caching**: Features are cached for 30 seconds to reduce API calls
- **Conditional Rendering**: Layers only render when enabled and data is available

## Future Enhancements

1. **Polygon Selection**: Allow drawing custom polygons instead of just rectangles
2. **Multiple Selection**: Select and highlight multiple features at once
3. **Feature Info Panel**: Show detailed information about highlighted features
4. **Export Selected Area**: Export features within selected area to GeoJSON
5. **Undo/Redo**: Track changes and allow undo/redo operations
