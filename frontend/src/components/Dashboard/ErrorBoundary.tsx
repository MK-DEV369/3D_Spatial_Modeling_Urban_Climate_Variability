import React from 'react'

type State = {
    hasError: boolean
    error?: any
}

class ErrorBoundary extends React.Component<
    { children: React.ReactNode },
    State
> {
    constructor(props: { children: React.ReactNode }) {
        super(props)
        this.state = { hasError: false }
    }

    static getDerivedStateFromError(error: any) {
        return { hasError: true, error }
    }

    componentDidCatch(error: any, errorInfo: any) {
        console.group('%c🧭 Map ErrorBoundary Triggered', 'color: red; font-weight: bold')

        console.error('❌ Error object:', error)
        console.error('🧩 Component stack:', errorInfo?.componentStack)

        // ---- Cesium / WebGL specific checks ----
        try {
            const cesium = (window as any).Cesium

            if (!cesium) {
                console.warn('⚠️ Cesium not found on window')
            } else {
                console.log('✅ Cesium version:', cesium.VERSION)

                // WebGL context check
                const canvas = document.querySelector('canvas')
                if (canvas) {
                    const gl =
                        canvas.getContext('webgl') ||
                        canvas.getContext('experimental-webgl')

                    if (!gl) {
                        console.error('❌ WebGL context NOT available')
                    } else {
                if (canvas) {
                    const gl =
                        canvas.getContext('webgl') ||
                        canvas.getContext('experimental-webgl')

                    if (!gl) {
                        console.error('❌ WebGL context NOT available')
                    } else if ('getParameter' in gl) {
                        console.log('✅ WebGL context OK')
                        console.log(
                            'GPU:',
                            gl.getParameter(gl.RENDERER),
                            gl.getParameter(gl.VENDOR)
                        )
                    } else {
                        console.warn('⚠️ Context is not WebGLRenderingContext')
                    }
                } else {
                    console.warn('⚠️ No canvas found (Viewer may not have mounted)')
                }
                    }
                }
            }
        } catch (e) {
            console.error('❌ Error while probing Cesium/WebGL:', e)
        }

        // ---- Geometry / memory hints ----
        if (error?.message) {
            if (error.message.toLowerCase().includes('memory')) {
                console.warn('🧠 Possible OUT-OF-MEMORY error')
            }
            if (error.message.toLowerCase().includes('primitive')) {
                console.warn('🧱 Primitive / geometry overload suspected')
            }
            if (error.message.toLowerCase().includes('destroy')) {
                console.warn('💥 Scene or Viewer destroyed during render')
            }
        }

        console.groupEnd()
    }

    render() {
        if (this.state.hasError) {
            return (
                <div className="w-full h-full flex flex-col items-center justify-center bg-gray-900 text-red-500">
                    <div className="text-lg font-semibold">
                        Map rendering failed
                    </div>
                    <div className="text-sm text-gray-400 mt-2 text-center max-w-md">
                        This usually happens due to very large datasets, invalid geometry,
                        or GPU memory limits.
                    </div>
                    <div className="text-xs text-gray-500 mt-4">
                        Open the browser console for Cesium diagnostics.
                    </div>
                </div>
            )
        }

        return this.props.children
    }
}

export default ErrorBoundary
