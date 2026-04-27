import { Route, Routes } from "react-router-dom";
import { NavBar } from "./components/NavBar";
import { HomePage } from "./pages/HomePage";
import { EarlyStopPage } from "./pages/EarlyStopPage";
import { BestOfNPage } from "./pages/BestOfNPage";
import { TimelinePage } from "./pages/TimelinePage";
import { DataPage } from "./pages/DataPage";

export default function App() {
  return (
    <div className="min-h-screen bg-transparent text-slate-900">
      <NavBar />
      <main className="mx-auto max-w-7xl px-4 py-8">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/early-stop" element={<EarlyStopPage />} />
          <Route path="/best-of-n" element={<BestOfNPage />} />
          <Route path="/timeline" element={<TimelinePage />} />
          <Route path="/data" element={<DataPage />} />
        </Routes>
      </main>
    </div>
  );
}
