import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import DashboardPage from './pages/DashboardPage'
import DatasetsPage from './pages/DatasetsPage'
import AnalysisPage from './pages/AnalysisPage'
import ResultsPage from './pages/ResultsPage'
import MarketDetailPage from './pages/MarketDetailPage'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/" element={<DashboardPage />} />
          <Route path="/datasets" element={<DatasetsPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/results" element={<ResultsPage />} />
          <Route path="/market/:tokenId" element={<MarketDetailPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}
