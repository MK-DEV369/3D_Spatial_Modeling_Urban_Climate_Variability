import { ReactNode } from 'react'
import { useLocation, Outlet } from 'react-router-dom'
import CardNav from '../CardNav'
import type { CardNavItem } from '../CardNav'

interface LayoutProps {
  children?: ReactNode
}

export default function Layout({ children }: LayoutProps) {
  const location = useLocation()

  // Create a simple logo data URL or use a path to your logo
  const logoDataUrl = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='45' fill='%2360A5FA'/%3E%3Ctext x='50' y='65' font-size='40' font-weight='bold' text-anchor='middle' fill='white'%3EUC%3C/text%3E%3C/svg%3E"

  const navItems: CardNavItem[] = [
    {
      label: "Home",
      bgColor: "#0F172A",
      textColor: "#60A5FA",
      links: [
        { label: "Welcome", href: "/", ariaLabel: "Go to Homepage" },
        { label: "About Project", href: "/#about", ariaLabel: "About the Project" }
      ]
    },
    {
      label: "Dashboard",
      bgColor: "#1E293B",
      textColor: "#60A5FA",
      links: [
        { label: "Metrics", href: "/dashboard", ariaLabel: "View Dashboard Metrics" },
        { label: "Analytics", href: "/dashboard#analytics", ariaLabel: "View Analytics" }
      ]
    },
    {
      label: "3D Viewer",
      bgColor: "#334155",
      textColor: "#60A5FA",
      links: [
        { label: "City View", href: "/viewer3d", ariaLabel: "View 3D City Model" },
        { label: "Buildings", href: "/viewer3d#buildings", ariaLabel: "View Buildings" }
      ]
    },
    {
      label: "Scenarios",
      bgColor: "#475569",
      textColor: "#60A5FA",
      links: [
        { label: "Create New", href: "/scenario", ariaLabel: "Create New Scenario" },
        { label: "Saved Scenarios", href: "/scenario#saved", ariaLabel: "View Saved Scenarios" }
      ]
    }
  ]

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      <CardNav
        logo={logoDataUrl}
        logoAlt="Urban Climate Modeling Logo"
        items={navItems}
        baseColor="#ffffff"
        menuColor="#1E293B"
        buttonBgColor="#1E293B"
        buttonTextColor="#60A5FA"
        ease="power3.out"
      />
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {children || <Outlet />}
      </main>
    </div>
  )
}

