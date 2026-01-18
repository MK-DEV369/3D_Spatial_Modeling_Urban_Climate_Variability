import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Homepage from './components/Homepage/Homepage'
import Dashboard from './components/Dashboard/Dashboard'
import ScenarioBuilder from './components/ScenarioBuilder/ScenarioBuilder'
import Layout from './components/common/Layout'

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Homepage />} />
        <Route element={<Layout />}>
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/scenario" element={<ScenarioBuilder />} />
        </Route>
      </Routes>
    </Router>
  )
}

export default App

