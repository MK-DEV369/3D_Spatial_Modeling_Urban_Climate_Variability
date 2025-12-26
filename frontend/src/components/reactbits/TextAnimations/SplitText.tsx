import { ReactNode, useEffect, useState } from 'react'

interface SplitTextProps {
  children: string
  splitBy?: 'word' | 'char' | 'line'
  stagger?: number
  direction?: 'up' | 'down' | 'left' | 'right'
  trigger?: 'onMount' | 'onScroll' | 'onHover'
  duration?: number
  className?: string
}

const directionClasses = {
  up: 'translate-y-4',
  down: '-translate-y-4',
  left: 'translate-x-4',
  right: '-translate-x-4',
}

export default function SplitText({
  children,
  splitBy = 'word',
  stagger = 50,
  direction = 'up',
  trigger = 'onMount',
  duration = 600,
  className = '',
}: SplitTextProps) {
  const [isVisible, setIsVisible] = useState(trigger === 'onMount')
  const [isHovered, setIsHovered] = useState(false)

  useEffect(() => {
    if (trigger === 'onMount') {
      setIsVisible(true)
    }
  }, [trigger])

  const handleScroll = () => {
    if (trigger === 'onScroll') {
      setIsVisible(true)
    }
  }

  useEffect(() => {
    if (trigger === 'onScroll') {
      window.addEventListener('scroll', handleScroll)
      handleScroll() // Check initial position
      return () => window.removeEventListener('scroll', handleScroll)
    }
  }, [trigger])

  const splitContent = () => {
    if (splitBy === 'char') {
      return children.split('').filter((char) => char !== ' ')
    } else if (splitBy === 'word') {
      return children.split(' ')
    } else {
      return children.split('\n')
    }
  }

  const elements = splitContent()

  return (
    <div
      className={`inline-block ${className}`}
      onMouseEnter={() => trigger === 'onHover' && setIsHovered(true)}
    >
      {elements.map((element, index) => (
        <span
          key={index}
          className="inline-block"
          style={{
            opacity: isVisible || isHovered ? 1 : 0,
            transform: isVisible || isHovered ? 'translate(0, 0)' : directionClasses[direction],
            transition: `opacity ${duration}ms ease-out, transform ${duration}ms ease-out`,
            transitionDelay: `${index * stagger}ms`,
          }}
        >
          {element}
          {splitBy === 'word' && index < elements.length - 1 && '\u00A0'}
        </span>
      ))}
    </div>
  )
}

