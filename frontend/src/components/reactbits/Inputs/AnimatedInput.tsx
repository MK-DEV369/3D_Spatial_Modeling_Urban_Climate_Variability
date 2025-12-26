import { InputHTMLAttributes, useState } from 'react'

interface AnimatedInputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  duration?: number
  error?: string
}

export default function AnimatedInput({
  label,
  duration = 200,
  error,
  className = '',
  ...props
}: AnimatedInputProps) {
  const [isFocused, setIsFocused] = useState(false)

  return (
    <div className="w-full">
      {label && (
        <label
          htmlFor={props.id}
          className="block text-sm font-medium mb-2 text-gray-300"
        >
          {label}
        </label>
      )}
      <input
        {...props}
        onFocus={(e) => {
          setIsFocused(true)
          props.onFocus?.(e)
        }}
        onBlur={(e) => {
          setIsFocused(false)
          props.onBlur?.(e)
        }}
        className={`
          w-full px-4 py-2
          bg-gray-700 border rounded-md
          text-white
          focus:outline-none focus:ring-2 focus:ring-blue-500
          transition-all
          ${isFocused ? 'border-blue-500' : 'border-gray-600'}
          ${error ? 'border-red-500' : ''}
          ${className}
        `}
        style={{ transitionDuration: `${duration}ms` }}
      />
      {error && (
        <p className="mt-1 text-sm text-red-400 animate-fade-in">{error}</p>
      )}
    </div>
  )
}

