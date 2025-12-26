import { HoverCard } from '../reactbits'

interface MetricsCardProps {
  title: string
  value: string
  subtitle?: string
}

export default function MetricsCard({ title, value, subtitle }: MetricsCardProps) {
  return (
    <HoverCard hover shadow="lg" padding="lg" border>
      <h3 className="text-sm font-medium text-gray-400 mb-2">{title}</h3>
      <p className="text-3xl font-bold text-white mb-1">{value}</p>
      {subtitle && <p className="text-sm text-gray-400">{subtitle}</p>}
    </HoverCard>
  )
}

