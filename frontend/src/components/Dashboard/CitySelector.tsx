import { City } from '../../types/city'

interface CitySelectorProps {
  cities: City[]
  selectedCityId: number | null
  onSelectCity: (cityId: number) => void
}

export default function CitySelector({ cities, selectedCityId, onSelectCity }: CitySelectorProps) {
  return (
    <div>
      <label htmlFor="city-select" className="block text-sm font-medium mb-2 text-gray-300">
        Select City
      </label>
      <select
        id="city-select"
        value={selectedCityId || ''}
        onChange={(e) => onSelectCity(Number(e.target.value))}
        className="block w-full px-4 py-2 bg-gray-800 border border-gray-700 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
      >
        <option value="">-- Select a city --</option>
        {cities.map((city) => (
          <option key={city.id} value={city.id}>
            {city.name}, {city.country}
          </option>
        ))}
      </select>
    </div>
  )
}

