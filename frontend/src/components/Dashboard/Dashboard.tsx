import { useState } from 'react'
import { useCities, useCityClimate, useCityTraffic, useCityPollution } from '../../hooks/useCities'
import CitySelector from './CitySelector'
import MetricsCard from './MetricsCard'
import ComparisonChart from './ComparisonChart'
import { FadeIn, SplitText } from '../reactbits'
import Iridescence from '../Iridescence'

export default function Dashboard() {
  const [selectedCityId, setSelectedCityId] = useState<number | null>(null)
  const { data: cities, isLoading: citiesLoading } = useCities()
  const { data: climateData } = useCityClimate(selectedCityId || 0)
  const { data: trafficData } = useCityTraffic(selectedCityId || 0)
  const { data: pollutionData } = useCityPollution(selectedCityId || 0)

  if (citiesLoading) {
    return (
      <div className="text-center py-12">
        <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <p className="mt-4 text-gray-400">Loading cities...</p>
      </div>
    )
  }

  const latestClimate = climateData?.[0]
  const latestTraffic = trafficData?.[0]
  const latestPollution = pollutionData?.[0]

  return (
    <div className="relative min-h-screen">
      <div className="fixed inset-0 z-0">
        <Iridescence />
      </div>
      <div className="relative z-10 px-4 py-6">
      <FadeIn duration={800} delay={0}>
        <SplitText
          splitBy="word"
          stagger={50}
          direction="up"
          trigger="onMount"
          duration={600}
          className="text-3xl font-bold mb-6"
        >
          Urban Climate Dashboard
        </SplitText>
      </FadeIn>
      
      <FadeIn duration={800} delay={200}>
        <div className="mb-6">
          <CitySelector
            cities={cities || []}
            selectedCityId={selectedCityId}
            onSelectCity={setSelectedCityId}
          />
        </div>
      </FadeIn>

      {selectedCityId && (
        <>
          <FadeIn duration={800} delay={400}>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-6">
              {latestClimate && (
                <MetricsCard
                  title="Temperature"
                  value={`${latestClimate.temperature.toFixed(1)}°C`}
                  subtitle={`Humidity: ${latestClimate.humidity?.toFixed(1) || 'N/A'}%`}
                />
              )}
              
              {latestTraffic && (
                <MetricsCard
                  title="Traffic"
                  value={latestTraffic.congestion_level || 'N/A'}
                  subtitle={`Speed: ${latestTraffic.speed?.toFixed(1) || 'N/A'} km/h`}
                />
              )}
              
              {latestPollution && (
                <MetricsCard
                  title="Air Quality"
                  value={`AQI: ${latestPollution.aqi || 'N/A'}`}
                  subtitle={`PM2.5: ${latestPollution.pm25?.toFixed(1) || 'N/A'} µg/m³`}
                />
              )}
            </div>
          </FadeIn>

          {climateData && climateData.length > 0 && (
            <FadeIn duration={800} delay={600}>
              <ComparisonChart
                title="Climate Trends"
                data={climateData}
                dataKey="temperature"
                xKey="timestamp"
                label="Temperature (°C)"
              />
            </FadeIn>
          )}
        </>
      )}

      {!selectedCityId && (
        <FadeIn duration={800} delay={400}>
          <div className="text-center py-12 text-gray-400">
            Select a city to view metrics
          </div>
        </FadeIn>
      )}
    </div>
    </div>
  )
}

