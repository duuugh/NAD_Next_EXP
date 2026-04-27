import { NavLink } from "react-router-dom";

const tabs = [
  { to: "/", label: "首页" },
  { to: "/early-stop", label: "Early Stop", tint: "emerald" },
  { to: "/best-of-n", label: "Best-of-N", tint: "blue" },
  { to: "/timeline", label: "方法演化" },
  { to: "/data", label: "数据与下载" },
];

export function NavBar() {
  return (
    <header className="sticky top-0 z-20 border-b border-slate-200 bg-white/90 backdrop-blur-xl">
      <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-3 md:flex-row md:items-center md:justify-between">
        <div>
          <div className="text-sm font-semibold tracking-[0.2em] text-slate-500">NAD_NEXT RESEARCH</div>
          <div className="text-lg font-semibold text-slate-950">Best-of-N × Early-Stop Showcase</div>
        </div>
        <nav className="flex flex-wrap gap-2 text-sm">
          {tabs.map((tab) => (
            <NavLink
              key={tab.to}
              to={tab.to}
              className={({ isActive }) =>
                [
                  "rounded-full px-4 py-2 transition",
                  isActive ? "bg-slate-950 text-white shadow-sm" : "bg-white text-slate-600 hover:bg-slate-100",
                ].join(" ")
              }
            >
              {tab.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}
