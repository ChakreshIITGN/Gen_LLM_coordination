import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Layout from './components/Layout';
import Landing from './pages/Landing';
import ExperimentList from './pages/ExperimentList';
import ExperimentDetail from './pages/ExperimentDetail';
import ExperimentComparison from './pages/ExperimentComparison';
import Analysis from './pages/Analysis';
import NewExperiment from './pages/NewExperiment';
import Setup from './pages/Setup';
import WorldExplainer from './pages/WorldExplainer';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<Landing />} />
          <Route path="/world" element={<WorldExplainer />} />
          <Route path="/experiments" element={<ExperimentList />} />
          <Route path="/experiment/:tag/:timestamp" element={<ExperimentDetail />} />
          <Route path="/compare" element={<ExperimentComparison />} />
          <Route path="/analysis" element={<Analysis />} />
          <Route path="/new" element={<NewExperiment />} />
          <Route path="/setup" element={<Setup />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
