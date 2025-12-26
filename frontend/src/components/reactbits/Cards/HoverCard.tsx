import { ReactNode, useState } from 'react'

interface HoverCardProps {
  children: ReactNode
  hover?: boolean
  shadow?: 'sm' | 'md' | 'lg' | 'xl'
  border?: boolean
  padding?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
  onClick?: () => void
}

const shadowClasses = {
  sm: 'shadow-sm',
  md: 'shadow-md',
  lg: 'shadow-lg',
  xl: 'shadow-xl',
}

const paddingClasses = {
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-6',
  xl: 'p-8',
}

export default function HoverCard({
  children,
  hover = true,
  shadow = 'md',
  border = true,
  padding = 'lg',
  className = '',
  onClick,
}: HoverCardProps) {
  const [isHovered, setIsHovered] = useState(false)

  return (
    <div
      className={`
        bg-gray-800 rounded-lg
        ${border ? 'border border-gray-700' : ''}
        ${shadowClasses[shadow]}
        ${paddingClasses[padding]}
        ${hover ? 'transition-all duration-300' : ''}
        ${isHovered && hover ? 'transform scale-105 border-gray-600 shadow-xl' : ''}
        ${onClick ? 'cursor-pointer' : ''}
        ${className}
      `}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={onClick}
    >
      {children}
    </div>
  )
}

