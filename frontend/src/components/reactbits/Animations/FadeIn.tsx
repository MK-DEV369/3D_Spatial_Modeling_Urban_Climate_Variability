import { ReactNode, useEffect, useState } from 'react'

interface FadeInProps {
  children: ReactNode
  duration?: number
  delay?: number
  easing?: string
  className?: string
  style?: React.CSSProperties
}

export default function FadeIn({
  children,
  duration = 600,
  delay = 0,
  easing = 'ease-out',
  className = '',
  style = {},
}: FadeInProps) {
  const [isVisible, setIsVisible] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsVisible(true)
    }, delay)

    return () => clearTimeout(timer)
  }, [delay])

  return (
    <div
      className={className}
      style={{
        opacity: isVisible ? 1 : 0,
        transition: `opacity ${duration}ms ${easing}`,
        ...style,
      }}
    >
      {children}
    </div>
  )
}

