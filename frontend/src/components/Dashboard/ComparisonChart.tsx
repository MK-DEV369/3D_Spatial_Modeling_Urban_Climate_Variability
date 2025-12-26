import { useMemo } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import { HoverCard } from '../reactbits'
import { ClimateData } from '../../types/climate'

interface ComparisonChartProps {
  title: string
  data: ClimateData[]
  dataKey: keyof ClimateData
  xKey: keyof ClimateData
  label: string
}

export default function ComparisonChart({
  title,
  data,
  dataKey,
  xKey,
  label,
}: ComparisonChartProps) {
  const chartData = useMemo(() => {
    return data
      .slice()
      .reverse()
      .slice(0, 30) // Show last 30 data points
      .map((item) => ({
        date: new Date(item[xKey as string]).toLocaleDateString('en-US', {
          month: 'short',
          day: 'numeric',
        }),
        value: item[dataKey] as number,
        timestamp: item[xKey as string],
      }))
  }, [data, dataKey, xKey])

  return (
    <HoverCard hover shadow="lg" padding="lg" border className="mb-6">
      <h3 className="text-xl font-semibold mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={300}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis
            dataKey="date"
            stroke="#9CA3AF"
            style={{ fontSize: '12px' }}
          />
          <YAxis
            stroke="#9CA3AF"
            style={{ fontSize: '12px' }}
            label={{ value: label, angle: -90, position: 'insideLeft', style: { fill: '#9CA3AF' } }}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1F2937',
              border: '1px solid #374151',
              borderRadius: '8px',
            }}
            labelStyle={{ color: '#F3F4F6' }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="value"
            stroke="#3B82F6"
            strokeWidth={2}
            dot={{ fill: '#3B82F6', r: 3 }}
            activeDot={{ r: 5 }}
            name={label}
          />
        </LineChart>
      </ResponsiveContainer>
    </HoverCard>
  )
}

