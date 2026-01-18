# Map Not Showing - Troubleshooting Guide

## Common Issues & Solutions

### 1. **OpenStreetMap iframe not loading**

#### Cause: CORS or mixed content issues
- Modern browsers block insecure content in iframes
- OpenStreetMap export embed requires proper parameters

#### Solution:
Ensure the iframe URL is correctly formatted:
```tsx
<iframe
  src={`https://www.openstreetmap.org/export/embed.html?bbox=${lon-0.1},${lat-0.1},${lon+0.1},${lat+0.1}&layer=mapnik&marker=${lat},${lon}`}
  className="w-full h-full"
  style={{ border: 0 }}
  title="Map"
  allow="geolocation"
/>
```

**Important**: Note the parameter order for bbox:
- Format: `bbox=min_lon,min_lat,max_lon,max_lat`
- Example: `bbox=77.4946,12.8716,77.6946,13.0716`

### 2. **City data not loading (no selectedCity)**

#### Cause: API not returning city data or cities array is empty

#### Check:
```tsx
console.log('Cities:', cities)
console.log('Selected City ID:', selectedCityId)
console.log('Selected City:', selectedCity)
```

#### Solution:
- Ensure backend is running: `http://localhost:8000/api/cities/`
- Check API response format (should be array or `{results: []}`)
- Verify cities have `latitude` and `longitude` fields

### 3. **Iframe showing blank/white screen**

#### Causes:
- Invalid coordinates (latitude/longitude out of range)
- Bbox too large or too small
- Missing latitude/longitude in city data

#### Solution:
```tsx
// Validate coordinates before rendering
const selectedCity = cities?.find((city: any) => city.id === selectedCityId)

if (!selectedCity?.latitude || !selectedCity?.longitude) {
  return <div>Invalid city coordinates</div>
}

// Ensure bbox is reasonable size (0.1-0.2 degrees)
const bboxSize = 0.1
const iframeSrc = `https://www.openstreetmap.org/export/embed.html?bbox=${
  selectedCity.longitude - bboxSize
},${selectedCity.latitude - bboxSize},${
  selectedCity.longitude + bboxSize
},${selectedCity.latitude + bboxSize
}&layer=mapnik&marker=${selectedCity.latitude},${selectedCity.longitude}`
```

### 4. **Backend not running / API errors**

#### Check if backend is accessible:
```powershell
# Test API endpoint
Invoke-WebRequest http://localhost:8000/api/cities/ | Select-Object StatusCode, Content
```

#### Expected response:
```json
[
  {
    "id": 1,
    "name": "Bengaluru",
    "country": "India",
    "latitude": 12.9716,
    "longitude": 77.5946,
    "population": 12000000
  }
]
```

#### Solution:
- Start backend: `cd backend && python manage.py runserver`
- Check database has city data: `python manage.py shell`
```python
from core.models import City
print(City.objects.all())
```

### 5. **CORS errors in browser console**

#### Symptoms:
```
Access to fetch at 'http://localhost:8000/api/cities/' from origin 'http://localhost:5173' 
has been blocked by CORS policy
```

#### Solution:
Install and configure django-cors-headers in backend:

```python
# backend/requirements.txt
django-cors-headers==4.3.1

# backend/urban_climate/settings.py
INSTALLED_APPS = [
    ...
    'corsheaders',
]

MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # Add at top
    'django.middleware.common.CommonMiddleware',
    ...
]

CORS_ALLOWED_ORIGINS = [
    "http://localhost:5173",  # Vite dev server
    "http://localhost:3000",  # Alternative frontend port
]

# Or for development only:
CORS_ALLOW_ALL_ORIGINS = True
```

### 6. **Map container has no height**

#### Cause: Parent container not sized properly

#### Solution:
Ensure all parent containers have explicit height:
```tsx
<div className="fixed inset-0 flex">  {/* Full viewport height */}
  <div className="flex-1 relative">  {/* Takes remaining space */}
    <iframe className="w-full h-full" />  {/* 100% of parent */}
  </div>
</div>
```

### 7. **React Query not fetching data**

#### Check:
- QueryClient is provided at app root
- Browser DevTools > Network tab shows API calls
- React Query DevTools shows query state

#### Solution:
```tsx
// Ensure QueryClientProvider wraps app
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'

const queryClient = new QueryClient()

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        {/* routes */}
      </Router>
    </QueryClientProvider>
  )
}
```

### 8. **OpenStreetMap export not available**

#### Alternative: Use direct tile layer
If OSM export embed is blocked, switch to Leaflet:

```bash
npm install leaflet react-leaflet
npm install -D @types/leaflet
```

```tsx
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'
import 'leaflet/dist/leaflet.css'

<MapContainer
  center={[selectedCity.latitude, selectedCity.longitude]}
  zoom={13}
  style={{ width: '100%', height: '100%' }}
>
  <TileLayer
    attribution='&copy; <a href="https://www.openstreetmap.org/">OpenStreetMap</a>'
    url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
  />
  <Marker position={[selectedCity.latitude, selectedCity.longitude]}>
    <Popup>{selectedCity.name}</Popup>
  </Marker>
</MapContainer>
```

## Debugging Checklist

- [ ] Backend running on http://localhost:8000
- [ ] Cities API returns data: http://localhost:8000/api/cities/
- [ ] City has valid latitude/longitude fields
- [ ] CORS configured in Django
- [ ] Frontend API_BASE_URL points to backend
- [ ] Browser console shows no errors
- [ ] Network tab shows successful API calls
- [ ] selectedCityId is not null
- [ ] selectedCity object is found
- [ ] Iframe src URL is properly formatted
- [ ] Map container has explicit height

## Quick Fix Script

Run this in browser console to debug:
```javascript
// Check if cities loaded
console.log('Cities:', document.querySelector('select')?.options.length)

// Check selected city
const select = document.querySelector('select')
console.log('Selected:', select?.value)

// Check iframe
const iframe = document.querySelector('iframe')
console.log('Iframe src:', iframe?.src)

// Check API
fetch('http://localhost:8000/api/cities/')
  .then(r => r.json())
  .then(d => console.log('API Response:', d))
  .catch(e => console.error('API Error:', e))
```

## Still Not Working?

1. **Clear browser cache**: Ctrl+Shift+Delete
2. **Hard reload**: Ctrl+Shift+R
3. **Check browser console**: F12 > Console tab
4. **Check network requests**: F12 > Network tab
5. **Try different browser**: Test in Chrome/Firefox/Edge
6. **Check firewall**: Ensure localhost:8000 is not blocked

---

**Note**: The most common issue is backend not running or CORS not configured. Always check backend first!
