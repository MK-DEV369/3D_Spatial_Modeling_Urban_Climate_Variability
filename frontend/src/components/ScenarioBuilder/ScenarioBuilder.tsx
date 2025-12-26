import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useCities } from '../../hooks/useCities'
import api from '../../services/api'
import { Scenario } from '../../types/scenario'
import CitySelector from '../Dashboard/CitySelector'
import { AnimatedButton, AnimatedInput, HoverCard, FadeIn, SplitText } from '../reactbits'
import Iridescence from '../Iridescence'

export default function ScenarioBuilder() {
  const [selectedCityId, setSelectedCityId] = useState<number | null>(null)
  const [scenarioName, setScenarioName] = useState('')
  const [description, setDescription] = useState('')
  const [timeHorizon, setTimeHorizon] = useState<'1d' | '7d' | '30d' | '1y' | '5y' | '10y'>('1y')
  const [parameters, setParameters] = useState({
    vegetation_change: 0,
    building_density_change: 0,
  })

  const { data: cities } = useCities()
  const queryClient = useQueryClient()

  const { data: scenarios } = useQuery<Scenario[]>({
    queryKey: ['scenarios'],
    queryFn: async () => {
      const response = await api.get('/scenarios/')
      return response.data
    },
  })

  const createScenarioMutation = useMutation({
    mutationFn: async (data: {
      name: string
      description: string
      city: number
      time_horizon: string
      parameters: Record<string, any>
    }) => {
      const response = await api.post('/scenarios/', data)
      return response.data
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['scenarios'] })
      setScenarioName('')
      setDescription('')
      setParameters({ vegetation_change: 0, building_density_change: 0 })
    },
  })

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!selectedCityId) return

    createScenarioMutation.mutate({
      name: scenarioName,
      description,
      city: selectedCityId,
      time_horizon: timeHorizon,
      parameters,
    })
  }

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
          Scenario Builder
        </SplitText>
      </FadeIn>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <FadeIn duration={800} delay={200}>
          <HoverCard hover shadow="lg" padding="lg" border>
            <h2 className="text-xl font-semibold mb-4">Create New Scenario</h2>
            
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <CitySelector
                  cities={cities || []}
                  selectedCityId={selectedCityId}
                  onSelectCity={setSelectedCityId}
                />
              </div>

              <AnimatedInput
                id="scenario-name"
                type="text"
                label="Scenario Name"
                value={scenarioName}
                onChange={(e) => setScenarioName(e.target.value)}
                required
              />

              <div>
                <label htmlFor="description" className="block text-sm font-medium mb-2 text-gray-300">
                  Description
                </label>
                <textarea
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
                  rows={3}
                />
              </div>

              <div>
                <label htmlFor="time-horizon" className="block text-sm font-medium mb-2 text-gray-300">
                  Time Horizon
                </label>
                <select
                  id="time-horizon"
                  value={timeHorizon}
                  onChange={(e) => setTimeHorizon(e.target.value as any)}
                  className="w-full px-4 py-2 bg-gray-700 border border-gray-600 rounded-md text-white focus:outline-none focus:ring-2 focus:ring-blue-500 transition-all duration-200"
                >
                  <option value="1d">1 Day</option>
                  <option value="7d">7 Days</option>
                  <option value="30d">30 Days</option>
                  <option value="1y">1 Year</option>
                  <option value="5y">5 Years</option>
                  <option value="10y">10 Years</option>
                </select>
              </div>

              <AnimatedInput
                id="vegetation-change"
                type="number"
                label="Vegetation Change (%)"
                value={parameters.vegetation_change.toString()}
                onChange={(e) => setParameters({ ...parameters, vegetation_change: Number(e.target.value) })}
                step="0.1"
              />

              <AnimatedInput
                id="building-density"
                type="number"
                label="Building Density Change (%)"
                value={parameters.building_density_change.toString()}
                onChange={(e) => setParameters({ ...parameters, building_density_change: Number(e.target.value) })}
                step="0.1"
              />

              <AnimatedButton
                type="submit"
                disabled={!selectedCityId || createScenarioMutation.isPending}
                loading={createScenarioMutation.isPending}
                variant="primary"
                size="lg"
                className="w-full"
              >
                Create Scenario
              </AnimatedButton>
            </form>
          </HoverCard>
        </FadeIn>

        <FadeIn duration={800} delay={400}>
          <HoverCard hover shadow="lg" padding="lg" border>
            <h2 className="text-xl font-semibold mb-4">Saved Scenarios</h2>
            
            {scenarios && scenarios.length > 0 ? (
              <div className="space-y-3">
                {scenarios.map((scenario, index) => (
                  <FadeIn key={scenario.id} duration={600} delay={index * 100}>
                    <HoverCard
                      hover
                      shadow="sm"
                      padding="md"
                      border
                      className="cursor-pointer"
                    >
                      <h3 className="font-semibold text-lg">{scenario.name}</h3>
                      <p className="text-sm text-gray-400 mt-1">{scenario.city_name}</p>
                      <p className="text-sm text-gray-300 mt-2">{scenario.description}</p>
                      <div className="mt-2 flex gap-4 text-xs text-gray-400">
                        <span>Horizon: {scenario.time_horizon}</span>
                        <span>Created: {new Date(scenario.created_at).toLocaleDateString()}</span>
                      </div>
                    </HoverCard>
                  </FadeIn>
                ))}
              </div>
            ) : (
              <p className="text-gray-400">No scenarios created yet</p>
            )}
          </HoverCard>
        </FadeIn>
      </div>
    </div>
    </div>
  )
}

