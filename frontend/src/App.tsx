import { useState, useEffect, useCallback } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts'
import { Activity, Bot, TrendingUp, TrendingDown, DollarSign, BarChart3, Settings, Play, Square, RefreshCw, Plus, History, Zap, AlertTriangle, Brain, ArrowUpDown } from 'lucide-react'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

interface Trade {
  id: string; symbol: string; side: string; entry_time: string; exit_time: string | null
  entry_price: number; exit_price: number | null; quantity: number; margin: number
  leverage: number; tp_price: number; sl_price: number; pnl: number; pnl_pct: number
  status: string; confidence: number; exit_reason: string; is_custom: boolean
}
interface Signal {
  symbol: string; direction: string; confidence: number; price: number
  rsi: number; adx: number; volume_ratio: number
}
interface BotStats {
  is_running: boolean; total_trades: number; open_trades: number; wins: number
  losses: number; win_rate: number; total_pnl: number; avg_pnl: number
  testnet: boolean; model_accuracy: number; model_trained: boolean
}
interface ModelStatus {
  is_trained: boolean; accuracy: number; metrics: Record<string, number>
  training_status: { status: string; progress: string; metrics: Record<string, number> }
}
interface BacktestResult {
  symbol?: string; total_trades: number; wins: number; losses: number; win_rate: number
  total_pnl: number; avg_pnl: number; max_drawdown: number; profit_factor: number
  sharpe_ratio: number; trades: Array<Record<string, number | string>>; equity_curve: number[]
  pair_results?: Record<string, Record<string, number | string | unknown>>
  summary?: Record<string, number>
}

const PAIRS = [
  'XRPUSDT','DOGEUSDT','ADAUSDT','SOLUSDT','SHIBUSDT','PEPEUSDT','LINKUSDT',
  'MATICUSDT','AVAXUSDT','ARBUSDT','OPUSDT','SUIUSDT','APTUSDT','NEARUSDT',
  'FTMUSDT','DOTUSDT','ATOMUSDT'
]

type TabId = 'dashboard' | 'trades' | 'signals' | 'backtest' | 'settings'

function App() {
  const [tab, setTab] = useState<TabId>('dashboard')
  const [stats, setStats] = useState<BotStats | null>(null)
  const [trades, setTrades] = useState<Trade[]>([])
  const [signals, setSignals] = useState<Signal[]>([])
  const [modelStatus, setModelStatus] = useState<ModelStatus | null>(null)
  const [logs, setLogs] = useState<string[]>([])
  const [backtestResult, setBacktestResult] = useState<BacktestResult | null>(null)
  const [loading, setLoading] = useState<Record<string, boolean>>({})
  const [error, setError] = useState('')
  const [apiKey, setApiKey] = useState('')
  const [apiSecret, setApiSecret] = useState('')
  const [testnet, setTestnet] = useState(true)
  const [customSymbol, setCustomSymbol] = useState('SOLUSDT')
  const [customSide, setCustomSide] = useState('LONG')
  const [backtestSymbol, setBacktestSymbol] = useState('')

  const api = useCallback(async (path: string, options?: RequestInit) => {
    const res = await fetch(`${API_URL}${path}`, { headers: { 'Content-Type': 'application/json' }, ...options })
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }))
      throw new Error(err.detail || 'API Error')
    }
    return res.json()
  }, [])

  const refresh = useCallback(async () => {
    try {
      const [s, t, m, l] = await Promise.all([
        api('/api/bot/status'), api('/api/trades'), api('/api/model/status'), api('/api/logs?limit=50'),
      ])
      setStats(s); setTrades(t.trades); setModelStatus(m); setLogs(l.logs)
    } catch { /* ignore initial errors */ }
  }, [api])

  useEffect(() => { refresh(); const iv = setInterval(refresh, 15000); return () => clearInterval(iv) }, [refresh])

  const setL = (k: string, v: boolean) => setLoading(p => ({ ...p, [k]: v }))

  const trainModel = async () => {
    setL('train', true); setError('')
    try {
      await api('/api/model/train', { method: 'POST', body: JSON.stringify({ num_candles: 3000 }) })
      const poll = setInterval(async () => {
        const st = await api('/api/model/status')
        setModelStatus(st)
        if (st.training_status.status === 'completed' || st.training_status.status === 'error') {
          clearInterval(poll); setL('train', false); refresh()
        }
      }, 5000)
    } catch (e) { setError(String(e)); setL('train', false) }
  }

  const startBot = async () => {
    setL('bot', true); setError('')
    try { await api('/api/bot/start', { method: 'POST' }); await refresh() }
    catch (e) { setError(String(e)) } finally { setL('bot', false) }
  }
  const stopBot = async () => {
    setL('bot', true); setError('')
    try { await api('/api/bot/stop', { method: 'POST' }); await refresh() }
    catch (e) { setError(String(e)) } finally { setL('bot', false) }
  }
  const manualScan = async () => {
    setL('scan', true); setError('')
    try { await api('/api/bot/scan', { method: 'POST' }); await refresh() }
    catch (e) { setError(String(e)) } finally { setL('scan', false) }
  }
  const fetchSignals = async () => {
    setL('signals', true); setError('')
    try { const r = await api('/api/signals'); setSignals(r.signals) }
    catch (e) { setError(String(e)) } finally { setL('signals', false) }
  }
  const customTrade = async () => {
    setL('custom', true); setError('')
    try {
      await api('/api/trades/custom', { method: 'POST', body: JSON.stringify({ symbol: customSymbol, side: customSide }) })
      await refresh()
    } catch (e) { setError(String(e)) } finally { setL('custom', false) }
  }
  const closeTrade = async (id: string) => {
    try { await api(`/api/trades/${id}/close`, { method: 'POST' }); await refresh() }
    catch (e) { setError(String(e)) }
  }
  const runBacktest = async () => {
    setL('backtest', true); setError('')
    try {
      const body: Record<string, unknown> = { num_candles: 3000 }
      if (backtestSymbol) body.symbol = backtestSymbol
      const r = await api('/api/backtest', { method: 'POST', body: JSON.stringify(body) })
      setBacktestResult(r)
    } catch (e) { setError(String(e)) } finally { setL('backtest', false) }
  }
  const saveConfig = async () => {
    setL('config', true); setError('')
    try {
      await api('/api/config', {
        method: 'POST', body: JSON.stringify({
          api_key: apiKey, api_secret: apiSecret, testnet,
          margin_per_trade: 1.0, leverage: 20, tp_pct: 50.0, sl_pct: 50.0,
          min_signal_confidence: 0.60, max_open_trades: 5, pairs: PAIRS,
        })
      })
      await refresh()
    } catch (e) { setError(String(e)) } finally { setL('config', false) }
  }

  const TabBtn = ({ id, icon: Icon, label }: { id: string; icon: React.ComponentType<{ className?: string }>; label: string }) => (
    <button onClick={() => setTab(id as TabId)}
      className={`flex items-center gap-2 px-4 py-2.5 rounded-lg text-sm font-medium transition-all ${tab === id ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/25' : 'text-gray-400 hover:text-white hover:bg-white/5'}`}>
      <Icon className="w-4 h-4" />{label}
    </button>
  )

  return (
    <div className="min-h-screen bg-[hsl(222.2,84%,4.9%)]">
      {/* Header */}
      <header className="border-b border-gray-800 bg-gray-900/50 backdrop-blur sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4 py-3 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Bot className="w-8 h-8 text-blue-500" />
            <div>
              <h1 className="text-lg font-bold text-white">ML Crypto Trading Bot</h1>
              <p className="text-xs text-gray-500">Binance Futures {testnet ? 'TESTNET' : 'MAINNET'}</p>
            </div>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <span className={`text-xs font-medium ${testnet ? 'text-yellow-400' : 'text-green-400'}`}>{testnet ? 'TESTNET' : 'REAL'}</span>
              <button onClick={() => setTestnet(!testnet)} className={`relative w-11 h-6 rounded-full transition-colors ${testnet ? 'bg-yellow-600' : 'bg-green-600'}`}>
                <span className={`absolute top-0.5 w-5 h-5 bg-white rounded-full transition-transform ${testnet ? 'left-0.5' : 'translate-x-5 left-0.5'}`} />
              </button>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${stats?.is_running ? 'bg-green-500 animate-pulse-green' : 'bg-gray-600'}`} />
              <span className="text-xs text-gray-400">{stats?.is_running ? 'Running' : 'Stopped'}</span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="border-b border-gray-800 bg-gray-900/30">
        <div className="max-w-7xl mx-auto px-4 py-2 flex gap-1 overflow-x-auto">
          <TabBtn id="dashboard" icon={Activity} label="Dashboard" />
          <TabBtn id="trades" icon={History} label="Trades" />
          <TabBtn id="signals" icon={Zap} label="Signals" />
          <TabBtn id="backtest" icon={BarChart3} label="Backtest" />
          <TabBtn id="settings" icon={Settings} label="Settings" />
        </div>
      </nav>

      {error && (
        <div className="max-w-7xl mx-auto px-4 mt-4">
          <div className="bg-red-900/30 border border-red-700 rounded-lg p-3 flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
            <p className="text-sm text-red-300">{error}</p>
            <button onClick={() => setError('')} className="ml-auto text-red-400 hover:text-red-300 text-xs">Dismiss</button>
          </div>
        </div>
      )}

      <main className="max-w-7xl mx-auto px-4 py-6">
        {/* DASHBOARD */}
        {tab === 'dashboard' && (
          <div className="space-y-6">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <StatCard icon={<DollarSign className="w-5 h-5 text-blue-400" />} label="Total PnL" value={`$${stats?.total_pnl?.toFixed(4) ?? '0'}`} color={stats?.total_pnl && stats.total_pnl > 0 ? 'text-green-400' : 'text-red-400'} />
              <StatCard icon={<TrendingUp className="w-5 h-5 text-green-400" />} label="Win Rate" value={`${stats?.win_rate ?? 0}%`} color="text-gray-200" />
              <StatCard icon={<ArrowUpDown className="w-5 h-5 text-purple-400" />} label="Total Trades" value={`${stats?.total_trades ?? 0}`} color="text-gray-200" />
              <StatCard icon={<Brain className="w-5 h-5 text-cyan-400" />} label="Model Accuracy" value={`${((stats?.model_accuracy ?? 0) * 100).toFixed(1)}%`} color="text-gray-200" />
            </div>

            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><Bot className="w-5 h-5 text-blue-400" />Bot Control</h2>
              <div className="flex flex-wrap gap-3">
                {!stats?.is_running ? (
                  <button onClick={startBot} disabled={loading.bot || !modelStatus?.is_trained} className="flex items-center gap-2 px-6 py-2.5 bg-green-600 hover:bg-green-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg font-medium transition">
                    <Play className="w-4 h-4" />{loading.bot ? 'Starting...' : 'Start Bot'}
                  </button>
                ) : (
                  <button onClick={stopBot} disabled={loading.bot} className="flex items-center gap-2 px-6 py-2.5 bg-red-600 hover:bg-red-700 text-white rounded-lg font-medium transition">
                    <Square className="w-4 h-4" />{loading.bot ? 'Stopping...' : 'Stop Bot'}
                  </button>
                )}
                <button onClick={manualScan} disabled={loading.scan || !modelStatus?.is_trained} className="flex items-center gap-2 px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg font-medium transition">
                  <RefreshCw className={`w-4 h-4 ${loading.scan ? 'animate-spin' : ''}`} />Manual Scan
                </button>
                <button onClick={trainModel} disabled={loading.train} className="flex items-center gap-2 px-4 py-2.5 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 text-white rounded-lg font-medium transition">
                  <Brain className={`w-4 h-4 ${loading.train ? 'animate-spin' : ''}`} />
                  {loading.train ? (modelStatus?.training_status?.progress || 'Training...') : 'Train Model'}
                </button>
              </div>
              {!modelStatus?.is_trained && <p className="mt-3 text-sm text-yellow-400 flex items-center gap-1"><AlertTriangle className="w-4 h-4" />Model not trained yet. Click &quot;Train Model&quot; to start.</p>}
              {modelStatus?.training_status?.status === 'collecting_data' && <p className="mt-3 text-sm text-blue-400">{modelStatus.training_status.progress}</p>}
              {modelStatus?.training_status?.status === 'training' && <p className="mt-3 text-sm text-purple-400">{modelStatus.training_status.progress}</p>}
            </div>

            {modelStatus?.is_trained && modelStatus.metrics && (
              <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><Brain className="w-5 h-5 text-cyan-400" />Model Metrics</h2>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MiniStat label="CV Accuracy" value={`${((modelStatus.metrics.cv_accuracy ?? 0) * 100).toFixed(1)}%`} />
                  <MiniStat label="Trade Accuracy" value={`${((modelStatus.metrics.trade_accuracy ?? 0) * 100).toFixed(1)}%`} />
                  <MiniStat label="Total Samples" value={`${modelStatus.metrics.total_samples ?? 0}`} />
                  <MiniStat label="Features" value={`${modelStatus.metrics.feature_count ?? 0}`} />
                </div>
              </div>
            )}

            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-4">Open Trades ({trades.filter(t => t.status === 'OPEN').length})</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="text-gray-500 border-b border-gray-800">
                    <th className="text-left py-2 px-2">Symbol</th><th className="text-left py-2 px-2">Side</th>
                    <th className="text-right py-2 px-2">Entry</th><th className="text-right py-2 px-2">TP</th>
                    <th className="text-right py-2 px-2">SL</th><th className="text-right py-2 px-2">Conf</th>
                    <th className="text-right py-2 px-2">Action</th>
                  </tr></thead>
                  <tbody>
                    {trades.filter(t => t.status === 'OPEN').map(t => (
                      <tr key={t.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                        <td className="py-2 px-2 font-medium">{t.symbol}</td>
                        <td className={`py-2 px-2 font-medium ${t.side === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>{t.side}</td>
                        <td className="py-2 px-2 text-right">{t.entry_price.toFixed(6)}</td>
                        <td className="py-2 px-2 text-right text-green-400">{t.tp_price.toFixed(6)}</td>
                        <td className="py-2 px-2 text-right text-red-400">{t.sl_price.toFixed(6)}</td>
                        <td className="py-2 px-2 text-right">{(t.confidence * 100).toFixed(1)}%</td>
                        <td className="py-2 px-2 text-right">
                          <button onClick={() => closeTrade(t.id)} className="px-2 py-1 bg-red-600/20 text-red-400 rounded text-xs hover:bg-red-600/40">Close</button>
                        </td>
                      </tr>
                    ))}
                    {trades.filter(t => t.status === 'OPEN').length === 0 && <tr><td colSpan={7} className="py-8 text-center text-gray-600">No open trades</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>

            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-4">Bot Logs</h2>
              <div className="bg-black/50 rounded-lg p-3 max-h-64 overflow-y-auto font-mono text-xs space-y-0.5">
                {logs.length > 0 ? logs.map((l, i) => (
                  <div key={i} className={`${l.includes('ERROR') ? 'text-red-400' : l.includes('SIGNAL') ? 'text-yellow-400' : l.includes('OPENED') ? 'text-green-400' : l.includes('CLOSED') ? 'text-blue-400' : 'text-gray-400'}`}>{l}</div>
                )) : <div className="text-gray-600">No logs yet</div>}
              </div>
            </div>
          </div>
        )}

        {/* TRADES */}
        {tab === 'trades' && (
          <div className="space-y-6">
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><Plus className="w-5 h-5 text-blue-400" />Custom Trade</h2>
              <div className="flex flex-wrap gap-3 items-end">
                <div>
                  <label className="text-xs text-gray-500 block mb-1">Symbol</label>
                  <select value={customSymbol} onChange={e => setCustomSymbol(e.target.value)} className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white">
                    {PAIRS.map(p => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <div>
                  <label className="text-xs text-gray-500 block mb-1">Side</label>
                  <select value={customSide} onChange={e => setCustomSide(e.target.value)} className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white">
                    <option value="LONG">LONG</option><option value="SHORT">SHORT</option>
                  </select>
                </div>
                <button onClick={customTrade} disabled={loading.custom} className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white rounded-lg text-sm font-medium transition">
                  <Plus className="w-4 h-4" />{loading.custom ? 'Opening...' : 'Open Trade'}
                </button>
              </div>
              <p className="mt-2 text-xs text-gray-500">$1 margin / 20x leverage / 50% TP & SL</p>
            </div>

            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><History className="w-5 h-5 text-purple-400" />Trade History</h2>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead><tr className="text-gray-500 border-b border-gray-800">
                    <th className="text-left py-2 px-2">ID</th><th className="text-left py-2 px-2">Symbol</th><th className="text-left py-2 px-2">Side</th>
                    <th className="text-left py-2 px-2">Status</th><th className="text-right py-2 px-2">Entry</th><th className="text-right py-2 px-2">Exit</th>
                    <th className="text-right py-2 px-2">PnL</th><th className="text-right py-2 px-2">PnL%</th><th className="text-right py-2 px-2">Conf</th>
                    <th className="text-left py-2 px-2">Time</th><th className="text-left py-2 px-2">Reason</th>
                  </tr></thead>
                  <tbody>
                    {trades.map(t => (
                      <tr key={t.id} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                        <td className="py-2 px-2 font-mono text-xs text-gray-500">{t.id}</td>
                        <td className="py-2 px-2 font-medium">{t.symbol}</td>
                        <td className={`py-2 px-2 font-medium ${t.side === 'LONG' ? 'text-green-400' : 'text-red-400'}`}>{t.side}</td>
                        <td className="py-2 px-2"><span className={`px-2 py-0.5 rounded-full text-xs font-medium ${t.status === 'WIN' ? 'bg-green-900/50 text-green-400' : t.status === 'LOSS' ? 'bg-red-900/50 text-red-400' : 'bg-blue-900/50 text-blue-400'}`}>{t.status}</span></td>
                        <td className="py-2 px-2 text-right font-mono text-xs">{t.entry_price.toFixed(6)}</td>
                        <td className="py-2 px-2 text-right font-mono text-xs">{t.exit_price?.toFixed(6) ?? '-'}</td>
                        <td className={`py-2 px-2 text-right font-medium ${t.pnl > 0 ? 'text-green-400' : t.pnl < 0 ? 'text-red-400' : ''}`}>${t.pnl.toFixed(4)}</td>
                        <td className={`py-2 px-2 text-right ${t.pnl_pct > 0 ? 'text-green-400' : t.pnl_pct < 0 ? 'text-red-400' : ''}`}>{t.pnl_pct.toFixed(1)}%</td>
                        <td className="py-2 px-2 text-right">{(t.confidence * 100).toFixed(0)}%</td>
                        <td className="py-2 px-2 text-xs text-gray-500">{t.entry_time?.slice(0, 16) ?? ''}</td>
                        <td className="py-2 px-2 text-xs text-gray-500">{t.exit_reason || (t.is_custom ? 'custom' : '-')}</td>
                      </tr>
                    ))}
                    {trades.length === 0 && <tr><td colSpan={11} className="py-8 text-center text-gray-600">No trades yet</td></tr>}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        )}

        {/* SIGNALS */}
        {tab === 'signals' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <h2 className="text-lg font-semibold">ML Signals</h2>
              <button onClick={fetchSignals} disabled={loading.signals} className="flex items-center gap-2 px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white rounded-lg text-sm font-medium transition">
                <RefreshCw className={`w-4 h-4 ${loading.signals ? 'animate-spin' : ''}`} />Refresh Signals
              </button>
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {signals.map(s => (
                <div key={s.symbol} className={`bg-gray-900/50 border rounded-xl p-4 ${s.direction === 'LONG' ? 'border-green-800' : s.direction === 'SHORT' ? 'border-red-800' : 'border-gray-800'}`}>
                  <div className="flex items-center justify-between mb-3">
                    <span className="font-bold text-white">{s.symbol}</span>
                    <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${s.direction === 'LONG' ? 'bg-green-900/50 text-green-400' : s.direction === 'SHORT' ? 'bg-red-900/50 text-red-400' : 'bg-gray-800 text-gray-400'}`}>
                      {s.direction === 'LONG' && <TrendingUp className="w-3 h-3 inline mr-1" />}
                      {s.direction === 'SHORT' && <TrendingDown className="w-3 h-3 inline mr-1" />}
                      {s.direction}
                    </span>
                  </div>
                  <div className="text-2xl font-bold text-white mb-2">${s.price.toFixed(s.price < 1 ? 8 : 4)}</div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="text-gray-500">Confidence <span className="text-white font-medium">{(s.confidence * 100).toFixed(1)}%</span></div>
                    <div className="text-gray-500">RSI <span className="text-white font-medium">{s.rsi}</span></div>
                    <div className="text-gray-500">ADX <span className="text-white font-medium">{s.adx}</span></div>
                    <div className="text-gray-500">Volume <span className="text-white font-medium">{s.volume_ratio}x</span></div>
                  </div>
                  <div className="mt-3 h-1.5 bg-gray-800 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${s.confidence > 0.7 ? 'bg-green-500' : s.confidence > 0.5 ? 'bg-yellow-500' : 'bg-gray-600'}`} style={{ width: `${s.confidence * 100}%` }} />
                  </div>
                </div>
              ))}
              {signals.length === 0 && <div className="col-span-3 text-center py-12 text-gray-600">{modelStatus?.is_trained ? 'Click "Refresh Signals" to load' : 'Train the model first'}</div>}
            </div>
          </div>
        )}

        {/* BACKTEST */}
        {tab === 'backtest' && (
          <div className="space-y-6">
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-4 flex items-center gap-2"><BarChart3 className="w-5 h-5 text-blue-400" />Backtest</h2>
              <div className="flex flex-wrap gap-3 items-end">
                <div>
                  <label className="text-xs text-gray-500 block mb-1">Symbol (empty = all)</label>
                  <select value={backtestSymbol} onChange={e => setBacktestSymbol(e.target.value)} className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-sm text-white">
                    <option value="">All Pairs</option>
                    {PAIRS.map(p => <option key={p} value={p}>{p}</option>)}
                  </select>
                </div>
                <button onClick={runBacktest} disabled={loading.backtest || !modelStatus?.is_trained} className="flex items-center gap-2 px-6 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg text-sm font-medium transition">
                  <BarChart3 className={`w-4 h-4 ${loading.backtest ? 'animate-spin' : ''}`} />{loading.backtest ? 'Running...' : 'Run Backtest'}
                </button>
              </div>
            </div>

            {backtestResult && (
              <>
                <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                  <h3 className="font-semibold mb-4">{backtestResult.summary ? 'Overall Results' : `Results: ${backtestResult.symbol}`}</h3>
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
                    {backtestResult.summary ? (<>
                      <MiniStat label="Total Trades" value={String(backtestResult.summary.total_trades ?? 0)} />
                      <MiniStat label="Win Rate" value={`${backtestResult.summary.overall_win_rate ?? 0}%`} />
                      <MiniStat label="Total PnL" value={`$${(backtestResult.summary.total_pnl ?? 0).toFixed(4)}`} />
                      <MiniStat label="Wins" value={String(backtestResult.summary.total_wins ?? 0)} />
                      <MiniStat label="Losses" value={String(backtestResult.summary.total_losses ?? 0)} />
                    </>) : (<>
                      <MiniStat label="Total Trades" value={String(backtestResult.total_trades)} />
                      <MiniStat label="Win Rate" value={`${backtestResult.win_rate}%`} />
                      <MiniStat label="Total PnL" value={`$${backtestResult.total_pnl.toFixed(4)}`} />
                      <MiniStat label="Max Drawdown" value={`${backtestResult.max_drawdown}%`} />
                      <MiniStat label="Profit Factor" value={String(backtestResult.profit_factor)} />
                    </>)}
                  </div>
                </div>

                {backtestResult.equity_curve && backtestResult.equity_curve.length > 0 && (
                  <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                    <h3 className="font-semibold mb-4">Equity Curve</h3>
                    <ResponsiveContainer width="100%" height={300}>
                      <LineChart data={backtestResult.equity_curve.map((v, i) => ({ idx: i, equity: v }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="idx" stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }} />
                        <Line type="monotone" dataKey="equity" stroke="#3B82F6" strokeWidth={2} dot={false} />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {backtestResult.trades && backtestResult.trades.length > 0 && (
                  <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                    <h3 className="font-semibold mb-4">Trade PnL Distribution</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <BarChart data={backtestResult.trades.map((t, i) => ({ idx: i, pnl: t.pnl as number }))}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="idx" stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <YAxis stroke="#6B7280" tick={{ fontSize: 11 }} />
                        <Tooltip contentStyle={{ background: '#1F2937', border: '1px solid #374151', borderRadius: '8px' }} />
                        <Bar dataKey="pnl">
                          {backtestResult.trades.map((t, i) => (
                            <Cell key={i} fill={(t.pnl as number) >= 0 ? '#22C55E' : '#EF4444'} />
                          ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {backtestResult.pair_results && (
                  <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
                    <h3 className="font-semibold mb-4">Per-Pair Results</h3>
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm">
                        <thead><tr className="text-gray-500 border-b border-gray-800">
                          <th className="text-left py-2 px-2">Pair</th><th className="text-right py-2 px-2">Trades</th>
                          <th className="text-right py-2 px-2">Win Rate</th><th className="text-right py-2 px-2">PnL</th>
                          <th className="text-right py-2 px-2">PF</th><th className="text-right py-2 px-2">Max DD</th>
                        </tr></thead>
                        <tbody>
                          {Object.entries(backtestResult.pair_results).map(([pair, r]) => {
                            const data = r as Record<string, number>
                            return (
                              <tr key={pair} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                                <td className="py-2 px-2 font-medium">{pair}</td>
                                <td className="py-2 px-2 text-right">{data.total_trades ?? 0}</td>
                                <td className="py-2 px-2 text-right">{data.win_rate ?? 0}%</td>
                                <td className={`py-2 px-2 text-right font-medium ${(data.total_pnl ?? 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>${(data.total_pnl ?? 0).toFixed(4)}</td>
                                <td className="py-2 px-2 text-right">{data.profit_factor ?? 0}</td>
                                <td className="py-2 px-2 text-right text-red-400">{data.max_drawdown ?? 0}%</td>
                              </tr>
                            )
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </>
            )}
          </div>
        )}

        {/* SETTINGS */}
        {tab === 'settings' && (
          <div className="space-y-6">
            <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-6">
              <h2 className="text-lg font-semibold mb-6 flex items-center gap-2"><Settings className="w-5 h-5 text-blue-400" />Configuration</h2>
              <div className="space-y-6">
                <div className="flex items-center justify-between p-4 bg-gray-800/50 rounded-lg">
                  <div><h3 className="font-medium">Network Mode</h3><p className="text-sm text-gray-500">Switch between Testnet and Real</p></div>
                  <div className="flex items-center gap-3">
                    <span className={`text-sm ${testnet ? 'text-gray-500' : 'text-green-400 font-medium'}`}>Real</span>
                    <button onClick={() => setTestnet(!testnet)} className={`relative w-14 h-7 rounded-full transition-colors ${testnet ? 'bg-yellow-600' : 'bg-green-600'}`}>
                      <span className={`absolute top-0.5 w-6 h-6 bg-white rounded-full transition-transform ${testnet ? 'left-0.5' : 'translate-x-7 left-0.5'}`} />
                    </button>
                    <span className={`text-sm ${testnet ? 'text-yellow-400 font-medium' : 'text-gray-500'}`}>Testnet</span>
                  </div>
                </div>
                {testnet && <div className="p-3 bg-yellow-900/20 border border-yellow-800 rounded-lg text-sm text-yellow-300">Testnet: https://testnet.binancefuture.com</div>}
                {!testnet && <div className="p-3 bg-red-900/20 border border-red-800 rounded-lg text-sm text-red-300">WARNING: Real trading mode!</div>}
                <div className="space-y-4">
                  <div><label className="text-sm text-gray-400 block mb-1">API Key</label>
                    <input type="password" value={apiKey} onChange={e => setApiKey(e.target.value)} placeholder="Enter Binance API Key" className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500" /></div>
                  <div><label className="text-sm text-gray-400 block mb-1">API Secret</label>
                    <input type="password" value={apiSecret} onChange={e => setApiSecret(e.target.value)} placeholder="Enter Binance API Secret" className="w-full bg-gray-800 border border-gray-700 rounded-lg px-4 py-2.5 text-sm text-white focus:outline-none focus:ring-2 focus:ring-blue-500" /></div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <InfoBox label="Margin/Trade" value="$1.00" /><InfoBox label="Leverage" value="20x" />
                  <InfoBox label="Take Profit" value="50%" /><InfoBox label="Stop Loss" value="50%" />
                </div>
                <div><h3 className="font-medium mb-2">Trading Pairs ({PAIRS.length})</h3>
                  <div className="flex flex-wrap gap-2">{PAIRS.map(p => <span key={p} className="px-2 py-1 bg-gray-800 rounded text-xs text-gray-300">{p}</span>)}</div>
                </div>
                <button onClick={saveConfig} disabled={loading.config} className="flex items-center gap-2 px-6 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-700 text-white rounded-lg font-medium transition">
                  {loading.config ? 'Saving...' : 'Save Configuration'}
                </button>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  )
}

function StatCard({ icon, label, value, color }: { icon: React.ReactNode; label: string; value: string; color: string }) {
  return (
    <div className="bg-gray-900/50 border border-gray-800 rounded-xl p-4">
      <div className="flex items-center gap-2 mb-2">{icon}<span className="text-xs text-gray-500">{label}</span></div>
      <div className={`text-2xl font-bold ${color}`}>{value}</div>
    </div>
  )
}

function MiniStat({ label, value }: { label: string; value: string }) {
  return <div className="bg-gray-800/50 rounded-lg p-3"><div className="text-xs text-gray-500 mb-1">{label}</div><div className="text-lg font-bold text-white">{value}</div></div>
}

function InfoBox({ label, value }: { label: string; value: string }) {
  return <div className="bg-gray-800/50 rounded-lg p-3 text-center"><div className="text-xs text-gray-500 mb-1">{label}</div><div className="text-lg font-bold text-blue-400">{value}</div></div>
}

export default App
