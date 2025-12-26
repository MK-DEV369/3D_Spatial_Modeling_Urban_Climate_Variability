# React Frontend - Urban Climate Modeling Dashboard

## Setup Instructions

1. Install dependencies:
```bash
npm install
```

2. Install ReactBits.dev components (optional):
```bash
npm install -g jsrepo
npx jsrepo init https://reactbits.dev/ts/tailwind/
```

3. Start development server:
```bash
npm run dev
```

## Project Structure

- `src/components/` - React components
  - `Dashboard/` - Dashboard components
  - `Viewer3D/` - Three.js 3D visualization
  - `ScenarioBuilder/` - Scenario creation UI
  - `common/` - Shared components
- `src/services/` - API service layer
- `src/hooks/` - React Query hooks
- `src/types/` - TypeScript type definitions

## Technologies

- React 18 + TypeScript
- Vite (build tool)
- React Router (routing)
- TanStack Query (API state management)
- Three.js + React Three Fiber (3D visualization)
- Tailwind CSS (styling)
- ReactBits.dev (animated components - optional)

## Environment Variables

Create `.env` file:
```
VITE_API_BASE_URL=http://localhost:8000/api
```

