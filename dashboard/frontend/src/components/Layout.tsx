import { useEffect, useState } from 'react';
import { NavLink, Outlet, Link } from 'react-router-dom';
import {
  HomeIcon,
  BeakerIcon,
  PlusCircleIcon,
  CogIcon,
  GlobeAltIcon,
  ChartBarIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { api } from '../lib/api';

const NAV_GROUPS = [
  {
    label: 'EXPLORE',
    items: [
      { to: '/', label: 'Home', icon: HomeIcon },
      { to: '/world', label: 'World', icon: GlobeAltIcon },
    ],
  },
  {
    label: 'EXPERIMENT',
    items: [
      { to: '/experiments', label: 'Runs', icon: BeakerIcon },
      { to: '/analysis', label: 'Analysis', icon: ChartBarIcon },
      { to: '/new', label: 'New Run', icon: PlusCircleIcon },
    ],
  },
  {
    label: 'SYSTEM',
    items: [
      { to: '/setup', label: 'Setup', icon: CogIcon },
    ],
  },
];

const STORAGE_KEY = 'sidebar-collapsed';

export default function Layout() {
  const [activeCount, setActiveCount] = useState(0);
  const [collapsed, setCollapsed] = useState(() => {
    try { return localStorage.getItem(STORAGE_KEY) === '1'; }
    catch { return false; }
  });

  function toggleCollapse() {
    setCollapsed((prev) => {
      const next = !prev;
      try { localStorage.setItem(STORAGE_KEY, next ? '1' : '0'); } catch {}
      return next;
    });
  }

  useEffect(() => {
    let cancelled = false;
    const poll = () => {
      api.getActiveJobs().then((r) => {
        if (!cancelled) setActiveCount(r.count);
      }).catch(() => {});
    };
    poll();
    const id = setInterval(poll, 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, []);

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <nav
        className={`shrink-0 bg-white/80 backdrop-blur border-r border-gray-200 flex flex-col py-6 transition-all duration-200 ${
          collapsed ? 'w-16 px-1.5' : 'w-56 px-3'
        }`}
      >
        {/* Title */}
        <Link to="/" className={`mb-8 ${collapsed ? 'px-0 flex justify-center' : 'px-3'}`}>
          {collapsed ? (
            <span className="text-base font-bold text-indigo-600">L</span>
          ) : (
            <>
              <h1 className="text-lg font-semibold text-gray-900 tracking-tight">
                LLM-ABM Lab
              </h1>
              <p className="text-[11px] text-gray-400 mt-0.5">Experiment Dashboard</p>
            </>
          )}
        </Link>

        {NAV_GROUPS.map((group) => (
          <div key={group.label} className="mb-4">
            {!collapsed && (
              <p className="text-[10px] font-semibold text-gray-400 uppercase tracking-widest px-3 mb-1.5">
                {group.label}
              </p>
            )}
            {group.items.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                end
                title={collapsed ? label : undefined}
                className={({ isActive }) =>
                  `flex items-center gap-2.5 py-2 rounded-lg text-sm font-medium transition-colors ${
                    collapsed ? 'justify-center px-0' : 'px-3'
                  } ${
                    isActive
                      ? 'bg-indigo-50 text-indigo-700'
                      : 'text-gray-600 hover:bg-gray-100 hover:text-gray-900'
                  }`
                }
              >
                <Icon className="w-5 h-5 shrink-0" />
                {!collapsed && label}
                {/* Active run badge */}
                {label === 'Runs' && activeCount > 0 && (
                  collapsed ? (
                    <span className="absolute top-0.5 right-0.5 w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                  ) : (
                    <span className="ml-auto flex items-center gap-1">
                      <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
                      <span className="text-xs text-indigo-600 font-semibold">{activeCount}</span>
                    </span>
                  )
                )}
              </NavLink>
            ))}
          </div>
        ))}

        {/* Collapse toggle at bottom */}
        <div className="mt-auto pt-4 border-t border-gray-100">
          <button
            onClick={toggleCollapse}
            className={`flex items-center gap-2 w-full py-2.5 rounded-lg text-gray-500 hover:bg-gray-100 hover:text-gray-700 transition-colors ${
              collapsed ? 'justify-center' : 'px-3'
            }`}
            title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          >
            {collapsed ? (
              <ChevronRightIcon className="w-5 h-5" />
            ) : (
              <>
                <ChevronLeftIcon className="w-5 h-5" />
                <span className="text-xs">Collapse</span>
              </>
            )}
          </button>
        </div>
      </nav>

      {/* Main content */}
      <main className="flex-1 overflow-y-auto p-6 lg:p-8">
        <Outlet />
      </main>
    </div>
  );
}
