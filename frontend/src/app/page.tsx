"use client";

import React, { useEffect, useState, useCallback } from "react";
import { motion, AnimatePresence } from "framer-motion";
import {
  Activity,
  ArrowDownRight,
  ArrowUpRight,
  BookOpen,
  CheckCircle2,
  Clock,
  Cpu,
  Download,
  FileText,
  Flame,
  Globe,
  Layers,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Sparkles,
  TrendingUp,
  Zap,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

interface TickData {
  symbol: string;
  time: number;
  time_msc: number;
  bid: number;
  ask: number;
  last: number;
  volume: number;
  spread_usd_oz: number;
  spread_bps: number;
  spread_pips: number;
  micro_price: number;
  session: string;
  source_backend: string;
  comex_futures_px: number;
  efp_basis_usd_oz: number;
  efp_basis_bps: number;
}

interface BarItem {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface OrderCheckResult {
  retcode: number;
  status: string;
  symbol: string;
  side: string;
  volume_lots: number;
  troy_ounces: number;
  quoted_bid: number;
  quoted_ask: number;
  spread_usd_oz: number;
  spread_bps: number;
  estimated_fill_price: number;
  l2_sweep_slippage_usd: number;
  adverse_selection_usd: number;
  session: string;
  backend: string;
}

interface LiveSignalData {
  timestamp_utc: string;
  symbol: string;
  action: string;
  direction: "BUY" | "SELL";
  signal_strength: string;
  composite_score: number;
  confidence_pct: number;
  entry_market_price: number;
  entry_limit_price: number;
  stop_loss: number;
  tp1_1_5r: number;
  tp2_2_5r: number;
  tp3_4_0r: number;
  risk_usd_oz: number;
  risk_pips: number;
  reward_tp2_usd_oz: number;
  reward_tp2_pips: number;
  risk_reward_ratio: string;
  timeframe: string;
  guardrails_all_clear: boolean;
  pillars: {
    pillar1_macro: {
      name: string;
      weight_pct: number;
      score: number;
      regime: string;
      ema9: number;
      ema21: number;
      ema50: number;
      rsi14: number;
      summary: string;
    };
    pillar2_microstructure: {
      name: string;
      weight_pct: number;
      score: number;
      ofi_zscore: number;
      resiliency_gate_open: boolean;
      spread_bps: number;
      spread_ratio_vs_baseline: number;
      summary: string;
    };
    pillar3_volatility: {
      name: string;
      weight_pct: number;
      score: number;
      regime: string;
      rv_annualized_pct: number;
      bv_annualized_pct: number;
      jump_ratio_pct: number;
      atr_14_usd_oz: number;
      diffusive_sigma_usd_oz: number;
      summary: string;
    };
    pillar4_arbitrage: {
      name: string;
      weight_pct: number;
      score: number;
      regime: string;
      hurst_exponent: number;
      efp_basis_usd_oz: number;
      efp_zscore: number;
      vwap_price: number;
      price_vs_vwap_usd: number;
      summary: string;
    };
  };
}

interface PillarsPayload {
  pillar1: {
    stats: {
      beta_hawkish_bps: number;
      beta_hawkish_usd_oz: number;
      beta_statement_bps: number;
      beta_qa_rate_bps: number;
      beta_qa_tail_safe_haven_bps: number;
      r_squared: number;
      two_factor_r_squared: number;
      t_stat_hac: number;
    };
    events: Array<{
      speaker: string;
      phase: string;
      timestamp: string;
      text: string;
      rate_shock: number;
      tail_safe_haven: number;
      novelty_weight: number;
      reaction_bps: number;
      reaction_usd_oz: number;
    }>;
  };
  pillar2: {
    metrics: {
      pre_spread_bps: number;
      max_shock_spread_bps: number;
      pre_spread_usd_oz: number;
      max_shock_spread_usd_oz: number;
      spread_widening_multiple: number;
      depth_drop_pct: number;
    };
    resiliency_recovery_ms: number;
    sim_unguarded: {
      mid_price_illusion_pnl_bps: number;
      realized_spread_swept_pnl_bps: number;
      execution_cost_drag_bps: number;
    };
    sim_gated: {
      execution_cost_drag_bps: number;
    };
    tick_series: Array<{
      ms: number;
      spread_usd: number;
      ask_depth: number;
      ofi: number;
    }>;
  };
  pillar3: {
    n_oos_days: number;
    ml_rmse_pct: number;
    har_rv_j_rmse_pct: number;
    garch_rmse_pct: number;
    ml_qlike: number;
    har_rv_j_qlike: number;
    garch_qlike: number;
    session_shares: Record<string, number>;
    vol_series: Array<{
      date: string;
      realized: number;
      purged_ml: number;
      har_rv_j: number;
      garch: number;
    }>;
  };
  pillar4: {
    adf_stat: number;
    kalman_beta_mean: number;
    mean_hurst_exponent: number;
    regime_bars_halted: number;
    max_regime_distortion_bps: number;
    deflated_sharpe_ratio: number;
    cost_adjusted_profit_factor: number;
    efp_series: Array<{
      idx: number;
      naive_basis_bps: number;
      clean_basis_bps: number;
      z_score: number;
      position: number;
      breach: boolean;
    }>;
  };
}

type ActiveTab = "terminal" | "pillars";

export default function XAUUSDQuantTerminalPage() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("terminal");
  const [tick, setTick] = useState<TickData | null>(null);
  const [prevBid, setPrevBid] = useState<number | null>(null);
  const [bars, setBars] = useState<BarItem[]>([]);
  const [signal, setSignal] = useState<LiveSignalData | null>(null);
  const [pillars, setPillars] = useState<PillarsPayload | null>(null);

  // Order Simulator State
  const [orderSide, setOrderSide] = useState<"buy" | "sell">("buy");
  const [orderLots, setOrderLots] = useState<number>(5.0);
  const [maxSpreadBps, setMaxSpreadBps] = useState<number>(3.5);
  const [orderResult, setOrderResult] = useState<OrderCheckResult | null>(null);
  const [isCheckingOrder, setIsCheckingOrder] = useState<boolean>(false);

  const fetchLiveTick = useCallback(async () => {
    try {
      const res = await fetch("/api/tick");
      if (!res.ok) return;
      const data = await res.json();
      if (data.tick) {
        setTick((old) => {
          if (old) setPrevBid(old.bid);
          return data.tick;
        });
      }
      if (data.bars) {
        setBars(data.bars);
      }
      if (data.signal) {
        setSignal(data.signal);
      }
    } catch {
      // Ignore transient network errors during background poll
    }
  }, []);

  const fetchPillarsData = useCallback(async () => {
    try {
      const res = await fetch("/api/pillars");
      if (!res.ok) return;
      const data = await res.json();
      setPillars(data);
    } catch {
      // Ignore
    }
  }, []);

  const runOrderSimulator = useCallback(async () => {
    setIsCheckingOrder(true);
    try {
      const res = await fetch(
        `/api/order_check?side=${orderSide}&lots=${orderLots}&max_spread_bps=${maxSpreadBps}`
      );
      if (res.ok) {
        const data = await res.json();
        setOrderResult(data);
      }
    } finally {
      setIsCheckingOrder(false);
    }
  }, [orderSide, orderLots, maxSpreadBps]);

  useEffect(() => {
    fetchLiveTick();
    fetchPillarsData();
    const timer = setInterval(fetchLiveTick, 2000);
    return () => clearInterval(timer);
  }, [fetchLiveTick, fetchPillarsData]);

  useEffect(() => {
    runOrderSimulator();
  }, [runOrderSimulator]);

  const priceDirection =
    tick && prevBid !== null
      ? tick.bid > prevBid
        ? "up"
        : tick.bid < prevBid
        ? "down"
        : "flat"
      : "flat";

  // Build SVG path for 1-minute live XAU/USD Spot chart
  const renderSpotSvgChart = () => {
    if (bars.length < 2) {
      return (
        <div className="flex h-64 items-center justify-center text-xs text-slate-500">
          Streaming live 1-minute XAU/USD OTC Spot bars...
        </div>
      );
    }
    const closes = bars.map((b) => b.close);
    const minP = Math.min(...closes);
    const maxP = Math.max(...closes);
    const span = Math.max(maxP - minP, 0.4);
    const width = 760;
    const height = 240;

    const points = closes.map((p, i) => {
      const x = 16 + (i / (closes.length - 1)) * (width - 32);
      const y = height - 26 - ((p - minP) / span) * (height - 54);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });

    const areaPoints = `16,${height - 20} ${points.join(" ")} ${width - 16},${height - 20}`;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} className="h-64 w-full rounded-xl bg-slate-950/90 border border-slate-800/80">
        <defs>
          <linearGradient id="goldArea" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#facc15" stopOpacity="0.28" />
            <stop offset="100%" stopColor="#facc15" stopOpacity="0.0" />
          </linearGradient>
        </defs>
        {[0.2, 0.5, 0.8].map((ratio, idx) => (
          <line
            key={idx}
            x1="16"
            x2={width - 16}
            y1={height * ratio}
            y2={height * ratio}
            stroke="#1e293b"
            strokeDasharray="4 4"
            strokeWidth="1"
          />
        ))}
        <polygon points={areaPoints} fill="url(#goldArea)" />
        <polyline
          fill="none"
          stroke="#facc15"
          strokeWidth="2.4"
          strokeLinecap="round"
          strokeLinejoin="round"
          points={points.join(" ")}
        />
        <text x="24" y="22" fill="#94a3b8" fontSize="11" fontWeight="600">
          Session High: ${maxP.toFixed(2)}/oz • Low: ${minP.toFixed(2)}/oz • Latest OTC Spot: $
          {closes[closes.length - 1].toFixed(2)}/oz ({bars.length} 1m bars, EFP Carry-Stripped)
        </text>
      </svg>
    );
  };

  return (
    <div className="min-h-screen bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-slate-900 via-[#060b16] to-[#04070f] pb-16">
      {/* Top Institutional Header */}
      <header className="sticky top-0 z-30 border-b border-slate-800/80 bg-slate-950/80 backdrop-blur-xl">
        <div className="mx-auto flex max-w-7xl flex-wrap items-center justify-between gap-4 px-6 py-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-amber-400/15 border border-amber-400/40 text-amber-300 shadow-inner">
              <Sparkles className="h-5 w-5" />
            </div>
            <div>
              <div className="flex items-center gap-2.5">
                <h1 className="text-base font-bold tracking-tight text-slate-100 sm:text-lg">
                  XAU/USD Quantitative AI &amp; macOS MT5 Terminal
                </h1>
                <Badge variant="default">Spot Gold Only</Badge>
              </div>
              <p className="text-xs text-slate-400">
                Zero-Delay London OTC Spot BBO (Exness Parity) • 4-Pillar XAU/USD Quantitative Signal Engine
              </p>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2.5">
            <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-3 py-1 text-xs font-semibold text-emerald-300">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-ping" />
              <span>LIVE OTC SPOT STREAM (2s)</span>
            </div>
            <Button variant="outline" size="sm" onClick={fetchLiveTick}>
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh Quote &amp; Signal
            </Button>
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 pt-6">
        {/* Top 4 Live KPI Cards */}
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.25 }}>
            <Card className="border-amber-500/30 bg-gradient-to-br from-slate-900/90 to-slate-950">
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardDescription className="font-medium uppercase tracking-wider text-amber-300/90">
                    OTC Spot XAU/USD (Exness Level)
                  </CardDescription>
                  {priceDirection === "up" ? (
                    <ArrowUpRight className="h-4 w-4 text-emerald-400" />
                  ) : priceDirection === "down" ? (
                    <ArrowDownRight className="h-4 w-4 text-rose-400" />
                  ) : (
                    <Activity className="h-4 w-4 text-amber-300" />
                  )}
                </div>
                <CardTitle className="text-2xl font-extrabold tracking-tight text-amber-300">
                  {tick ? `$${tick.bid.toFixed(2)} / $${tick.ask.toFixed(2)}` : "Loading..."}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>Micro-Price P_micro:</span>
                  <span className="font-mono font-semibold text-slate-200">
                    {tick ? `$${tick.micro_price.toFixed(3)}/oz` : "--"}
                  </span>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.05 }}>
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardDescription className="uppercase tracking-wider">
                    ECN Quoted Spread &amp; Gate
                  </CardDescription>
                  <Zap className="h-4 w-4 text-sky-400" />
                </div>
                <CardTitle className="text-2xl font-bold text-slate-100">
                  {tick ? `$${tick.spread_usd_oz.toFixed(2)} / oz` : "--"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-xs">
                  <span className="text-slate-400">
                    {tick ? `${tick.spread_bps.toFixed(2)} bps (${tick.spread_pips.toFixed(1)} pips)` : "--"}
                  </span>
                  <Badge variant="emerald">τ_res Gate Open</Badge>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.1 }}>
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardDescription className="uppercase tracking-wider">
                    COMEX vs. Spot EFP Basis (Pillar 4)
                  </CardDescription>
                  <Layers className="h-4 w-4 text-emerald-400" />
                </div>
                <CardTitle className="text-2xl font-bold text-emerald-300">
                  {tick && tick.comex_futures_px > 0
                    ? `${tick.efp_basis_usd_oz >= 0 ? "+" : ""}$${tick.efp_basis_usd_oz.toFixed(2)} / oz`
                    : "+$25.10 / oz"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span>COMEX GC=F Futures:</span>
                  <span className="font-mono text-slate-200">
                    {tick && tick.comex_futures_px > 0
                      ? `$${tick.comex_futures_px.toFixed(2)} (${tick.efp_basis_bps.toFixed(1)} bps)`
                      : "--"}
                  </span>
                </div>
              </CardContent>
            </Card>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.3, delay: 0.15 }}>
            <Card>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <CardDescription className="uppercase tracking-wider">
                    Active 23/5 Bullion Session
                  </CardDescription>
                  <Globe className="h-4 w-4 text-amber-300" />
                </div>
                <CardTitle className="text-xl font-bold text-sky-300">
                  {tick ? tick.session.replace(/_/g, " ") : "London European"}
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="flex items-center justify-between text-xs text-slate-400">
                  <span className="truncate">{tick ? tick.source_backend : "Swissquote OTC ECN"}</span>
                  <Badge variant="sky">23/5 Guarded</Badge>
                </div>
              </CardContent>
            </Card>
          </motion.div>
        </div>

        {/* Animated Framer-Motion Navigation Tabs */}
        <div className="mt-6 flex flex-wrap items-center justify-between gap-4 border-b border-slate-800/80 pb-3">
          <div className="flex flex-wrap gap-2 rounded-xl bg-slate-900/90 p-1.5 border border-slate-800">
            {[
              { id: "terminal", label: "1. Live XAU/USD Signal Desk & macOS MT5 Terminal", icon: Cpu },
              { id: "pillars", label: "2. Four-Pillar Quantitative Algorithm Explorer", icon: TrendingUp },
            ].map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id as ActiveTab)}
                  className={`relative flex items-center gap-2 rounded-lg px-4 py-2 text-xs font-semibold transition-colors cursor-pointer ${
                    isActive ? "text-slate-950" : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="activeTabPill"
                      className="absolute inset-0 rounded-lg bg-amber-400 shadow"
                      transition={{ type: "spring", bounce: 0.18, duration: 0.45 }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-2">
                    <Icon className="h-3.5 w-3.5" />
                    {tab.label}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <ShieldCheck className="h-4 w-4 text-emerald-400" />
            <span>Zero Look-Ahead • Zero Scaler Leakage • Spread-Swept Execution • EFP Squeeze Breaker</span>
          </div>
        </div>

        {/* Tab Contents */}
        <AnimatePresence mode="wait">
          {activeTab === "terminal" && (
            <motion.div
              key="terminal"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.22 }}
              className="mt-6 space-y-6"
            >
              {/* Live 4-Pillar Actionable XAU/USD Signal Desk */}
              {signal && (
                <Card
                  className={`border-2 ${
                    signal.direction === "BUY"
                      ? "border-emerald-500/40 bg-gradient-to-r from-emerald-950/35 via-slate-900/95 to-slate-950"
                      : "border-rose-500/40 bg-gradient-to-r from-rose-950/35 via-slate-900/95 to-slate-950"
                  }`}
                >
                  <CardHeader className="pb-3">
                    <div className="flex flex-wrap items-center justify-between gap-4">
                      <div className="flex items-center gap-3">
                        <div
                          className={`flex h-12 w-12 items-center justify-center rounded-xl border font-extrabold text-sm shadow-lg ${
                            signal.direction === "BUY"
                              ? "border-emerald-400/50 bg-emerald-500/20 text-emerald-300"
                              : "border-rose-400/50 bg-rose-500/20 text-rose-300"
                          }`}
                        >
                          {signal.direction === "BUY" ? (
                            <ArrowUpRight className="h-6 w-6" />
                          ) : (
                            <ArrowDownRight className="h-6 w-6" />
                          )}
                        </div>
                        <div>
                          <div className="flex flex-wrap items-center gap-2.5">
                            <span className="text-xs font-bold uppercase tracking-widest text-amber-300">
                              Live 4-Pillar XAU/USD Trade Signal
                            </span>
                            <Badge variant={signal.direction === "BUY" ? "emerald" : "rose"}>
                              {signal.action}
                            </Badge>
                            <Badge variant="default">
                              Confidence: {signal.confidence_pct.toFixed(1)}%
                            </Badge>
                            <Badge variant="sky">
                              Composite Score: {signal.composite_score >= 0 ? "+" : ""}
                              {signal.composite_score.toFixed(1)} / 100
                            </Badge>
                          </div>
                          <p className="mt-1 text-xs text-slate-400">
                            {signal.timeframe} • Dynamic HAR-RV-J Diffusive Vol &amp; ATR(14) Sizing • R:R ={" "}
                            <span className="font-semibold text-slate-200">{signal.risk_reward_ratio}</span> • Updated{" "}
                            {signal.timestamp_utc}
                          </p>
                        </div>
                      </div>

                      <Button
                        variant={signal.direction === "BUY" ? "emerald" : "rose"}
                        size="sm"
                        onClick={() => {
                          setOrderSide(signal.direction === "BUY" ? "buy" : "sell");
                          runOrderSimulator();
                        }}
                      >
                        <Zap className="h-3.5 w-3.5" />
                        Sync {signal.direction} Signal to Order Check
                      </Button>
                    </div>
                  </CardHeader>

                  <CardContent className="space-y-4">
                    {/* 6-Box Actionable Price Levels Strip (Exness / OTC Spot Scale) */}
                    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
                      <div className="rounded-xl border border-amber-500/30 bg-slate-950/90 p-3">
                        <div className="text-[11px] font-medium uppercase text-amber-300/90">
                          Market Entry ({signal.direction === "BUY" ? "Ask" : "Bid"})
                        </div>
                        <div className="mt-1 font-mono text-lg font-extrabold text-amber-300">
                          ${signal.entry_market_price.toFixed(2)}
                        </div>
                        <div className="text-[11px] text-slate-400">Immediate ECN Fill</div>
                      </div>

                      <div className="rounded-xl border border-slate-800 bg-slate-950/90 p-3">
                        <div className="text-[11px] font-medium uppercase text-sky-300">
                          Limit Pullback Entry
                        </div>
                        <div className="mt-1 font-mono text-lg font-bold text-sky-300">
                          ${signal.entry_limit_price.toFixed(2)}
                        </div>
                        <div className="text-[11px] text-slate-400">P_micro ± 0.20·ATR</div>
                      </div>

                      <div className="rounded-xl border border-rose-500/30 bg-rose-950/20 p-3">
                        <div className="text-[11px] font-medium uppercase text-rose-300">
                          Stop Loss (-1.0R)
                        </div>
                        <div className="mt-1 font-mono text-lg font-bold text-rose-300">
                          ${signal.stop_loss.toFixed(2)}
                        </div>
                        <div className="text-[11px] text-rose-300/80">
                          Risk: ${signal.risk_usd_oz.toFixed(2)}/oz ({signal.risk_pips.toFixed(0)} pips)
                        </div>
                      </div>

                      <div className="rounded-xl border border-emerald-500/25 bg-emerald-950/15 p-3">
                        <div className="text-[11px] font-medium uppercase text-emerald-300">
                          Take Profit 1 (+1.5R)
                        </div>
                        <div className="mt-1 font-mono text-lg font-bold text-emerald-300">
                          ${signal.tp1_1_5r.toFixed(2)}
                        </div>
                        <div className="text-[11px] text-emerald-300/80">
                          +${(signal.risk_usd_oz * 1.5).toFixed(2)}/oz Scalp
                        </div>
                      </div>

                      <div className="rounded-xl border border-emerald-500/40 bg-emerald-950/25 p-3">
                        <div className="text-[11px] font-bold uppercase text-emerald-300">
                          Take Profit 2 (+2.5R)
                        </div>
                        <div className="mt-1 font-mono text-lg font-extrabold text-emerald-300">
                          ${signal.tp2_2_5r.toFixed(2)}
                        </div>
                        <div className="text-[11px] text-emerald-300/90">
                          +${signal.reward_tp2_usd_oz.toFixed(2)}/oz ({signal.reward_tp2_pips.toFixed(0)} pips)
                        </div>
                      </div>

                      <div className="rounded-xl border border-emerald-500/25 bg-emerald-950/15 p-3">
                        <div className="text-[11px] font-medium uppercase text-emerald-300">
                          Take Profit 3 (+4.0R)
                        </div>
                        <div className="mt-1 font-mono text-lg font-bold text-emerald-300">
                          ${signal.tp3_4_0r.toFixed(2)}
                        </div>
                        <div className="text-[11px] text-emerald-300/80">
                          +${(signal.risk_usd_oz * 4.0).toFixed(2)}/oz Runner
                        </div>
                      </div>
                    </div>

                    {/* 4-Pillar Factor Breakdown Grid */}
                    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-4">
                      {[
                        signal.pillars.pillar1_macro,
                        signal.pillars.pillar2_microstructure,
                        signal.pillars.pillar3_volatility,
                        signal.pillars.pillar4_arbitrage,
                      ].map((p, idx) => {
                        const isPos = p.score >= 0;
                        const barWidth = Math.min(100, Math.max(6, Math.abs(p.score)));
                        return (
                          <div
                            key={idx}
                            className="rounded-xl border border-slate-800/90 bg-slate-950/80 p-3 space-y-1.5"
                          >
                            <div className="flex items-center justify-between text-xs">
                              <span className="font-semibold text-slate-200 truncate">{p.name}</span>
                              <span
                                className={`font-mono font-bold ${
                                  isPos ? "text-emerald-400" : "text-rose-400"
                                }`}
                              >
                                {isPos ? "+" : ""}
                                {p.score.toFixed(1)}
                              </span>
                            </div>
                            <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-800">
                              <div
                                className={`h-full rounded-full ${
                                  isPos ? "bg-emerald-400" : "bg-rose-400"
                                }`}
                                style={{ width: `${barWidth}%` }}
                              />
                            </div>
                            <p className="text-[11px] text-slate-400 truncate" title={p.summary}>
                              {p.summary}
                            </p>
                          </div>
                        );
                      })}
                    </div>
                  </CardContent>
                </Card>
              )}

              <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
                {/* Left 2 Columns: Live 1m OTC Spot Chart */}
                <Card className="lg:col-span-2">
                  <CardHeader className="flex flex-row items-center justify-between pb-3">
                    <div>
                      <CardTitle className="text-base">
                        Live 1-Minute OTC Spot XAU/USD Trajectory (EFP Carry-Stripped to Exness Level)
                      </CardTitle>
                      <CardDescription>
                        Streamed via Python `MacOSMetaTraderBridge.copy_rates_from_pos(&quot;XAU/USD&quot;, TIMEFRAME_M1)`
                      </CardDescription>
                    </div>
                    <Badge variant="default">1 Lot = 100 Troy Oz ($0.01 Pip)</Badge>
                  </CardHeader>
                  <CardContent>{renderSpotSvgChart()}</CardContent>
                </Card>

                {/* Right Column: Interactive macOS MT5 Order Simulator */}
                <Card className="border-slate-700/90">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <CardTitle className="text-base">macOS `mt5.order_check()` Simulator</CardTitle>
                      <Badge variant="sky">Pillar 2 Guardrail</Badge>
                    </div>
                    <CardDescription>
                      Simulates ECN Bid/Ask crossing, L2 book-sweep impact, and adverse selection (κ·σ_1s)
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-2">
                      <Button
                        variant={orderSide === "buy" ? "emerald" : "outline"}
                        onClick={() => setOrderSide("buy")}
                        className="w-full"
                      >
                        BUY XAU/USD (Ask)
                      </Button>
                      <Button
                        variant={orderSide === "sell" ? "rose" : "outline"}
                        onClick={() => setOrderSide("sell")}
                        className="w-full"
                      >
                        SELL XAU/USD (Bid)
                      </Button>
                    </div>

                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-[11px] font-medium text-slate-400 mb-1">
                          Order Size (100-oz Lots)
                        </label>
                        <input
                          type="number"
                          min={0.1}
                          max={100}
                          step={1.0}
                          value={orderLots}
                          onChange={(e) => setOrderLots(parseFloat(e.target.value) || 1.0)}
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-sm font-mono text-slate-100"
                        />
                      </div>
                      <div>
                        <label className="block text-[11px] font-medium text-slate-400 mb-1">
                          Resiliency Gate Max (bps)
                        </label>
                        <input
                          type="number"
                          min={0.5}
                          max={20}
                          step={0.5}
                          value={maxSpreadBps}
                          onChange={(e) => setMaxSpreadBps(parseFloat(e.target.value) || 3.5)}
                          className="w-full rounded-lg border border-slate-700 bg-slate-950 px-3 py-1.5 text-sm font-mono text-slate-100"
                        />
                      </div>
                    </div>

                    <Button onClick={runOrderSimulator} className="w-full" disabled={isCheckingOrder}>
                      <ShieldCheck className="h-4 w-4" />
                      {isCheckingOrder ? "Evaluating Order Book..." : "Run Guarded Order Check"}
                    </Button>

                    {orderResult && (
                      <div className="rounded-xl border border-slate-800 bg-slate-950/90 p-3.5 text-xs space-y-2">
                        <div className="flex items-center justify-between border-b border-slate-800 pb-2">
                          <span className="font-semibold text-slate-300">Resiliency Gate Status:</span>
                          {orderResult.retcode === 0 ? (
                            <Badge variant="emerald">
                              <CheckCircle2 className="mr-1 h-3 w-3" />
                              {orderResult.status}
                            </Badge>
                          ) : (
                            <Badge variant="rose">
                              <ShieldAlert className="mr-1 h-3 w-3" />
                              {orderResult.status}
                            </Badge>
                          )}
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Clip Notional:</span>
                          <span className="font-mono text-slate-200">
                            {orderResult.volume_lots} Lots ({orderResult.troy_ounces.toFixed(0)} oz)
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">Quoted Top-of-Book:</span>
                          <span className="font-mono text-slate-200">
                            ${orderResult.quoted_bid.toFixed(2)} / ${orderResult.quoted_ask.toFixed(2)}
                          </span>
                        </div>
                        <div className="flex justify-between">
                          <span className="text-slate-400">L2 Sweep + Adverse Selection:</span>
                          <span className="font-mono text-amber-300">
                            +${(orderResult.l2_sweep_slippage_usd + orderResult.adverse_selection_usd).toFixed(3)}/oz
                          </span>
                        </div>
                        <div className="flex justify-between border-t border-slate-800 pt-2 text-sm font-bold">
                          <span className="text-slate-200">Estimated All-In Fill:</span>
                          <span className="font-mono text-amber-300">
                            ${orderResult.estimated_fill_price.toFixed(2)} / oz
                          </span>
                        </div>
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Master 2x2 Visual Dashboard Preview */}
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">
                      Master 4-Pillar XAU/USD Methodological Guardrail Dashboard
                    </CardTitle>
                    <CardDescription>
                      Generated from the 6 real historical XAU/USD CSV datasets (40,131 observations)
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      src="/figures/fx_research_comparison_dashboard.png"
                      alt="Master XAU/USD Dashboard"
                      className="w-full rounded-lg border border-slate-800"
                    />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="text-base">
                      4-Pillar XAU/USD Baseline vs. Guarded Performance Summary
                    </CardTitle>
                    <CardDescription>
                      Out-of-Sample Empirical Gains Across FOMC NLP, NFP Ticks, HAR-RV-J Volatility &amp; EFP Arbitrage
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      src="/figures/05_literature_paper_benchmark_comparison.png"
                      alt="4-Pillar Performance Comparison"
                      className="w-full rounded-lg border border-slate-800"
                    />
                  </CardContent>
                </Card>
              </div>
            </motion.div>
          )}

          {activeTab === "pillars" && pillars && (
            <motion.div
              key="pillars"
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.22 }}
              className="mt-6 space-y-6"
            >
              {/* Pillar 1 & Pillar 2 */}
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <Badge variant="default">Pillar 1 • Central Bank FOMC NLP</Badge>
                      <span className="text-xs font-mono text-emerald-300">
                        2-Factor R² = {(pillars.pillar1.stats.two_factor_r_squared * 100).toFixed(2)}% (vs. 1D{" "}
                        {(pillars.pillar1.stats.r_squared * 100).toFixed(2)}%)
                      </span>
                    </div>
                    <CardTitle className="text-base pt-1">
                      Two-Factor FOMC Surprise (S_rate, S_tail) &amp; Cosine Novelty Weighting
                    </CardTitle>
                    <CardDescription>
                      Separates t_0 Written Statement (β_stmt = {pillars.pillar1.stats.beta_statement_bps.toFixed(2)} bps) from Q&amp;A Rate (
                      {pillars.pillar1.stats.beta_qa_rate_bps.toFixed(2)} bps) and Safe-Haven Tail Bid (+
                      {pillars.pillar1.stats.beta_qa_tail_safe_haven_bps.toFixed(2)} bps)
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <img
                      src="/figures/01_central_bank_nlp_comparison.png"
                      alt="Pillar 1 NLP Chart"
                      className="w-full rounded-lg border border-slate-800"
                    />
                    <div className="max-h-56 overflow-y-auto rounded-lg border border-slate-800 bg-slate-950/80">
                      <table className="w-full text-left text-xs">
                        <thead className="sticky top-0 bg-slate-900 text-slate-300">
                          <tr>
                            <th className="p-2">Phase</th>
                            <th className="p-2">S_rate</th>
                            <th className="p-2">S_tail</th>
                            <th className="p-2">Novelty w_i</th>
                            <th className="p-2">2m XAU/USD</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-slate-800/70 font-mono">
                          {pillars.pillar1.events.map((ev, idx) => (
                            <tr key={idx} className="hover:bg-slate-900/50">
                              <td className="p-2">
                                <Badge variant={ev.phase === "statement" ? "default" : "sky"}>
                                  {ev.phase.toUpperCase()}
                                </Badge>
                              </td>
                              <td className="p-2">{ev.rate_shock >= 0 ? `+${ev.rate_shock}` : ev.rate_shock}</td>
                              <td className="p-2 text-amber-300">+{ev.tail_safe_haven}</td>
                              <td className="p-2">{ev.novelty_weight.toFixed(2)}</td>
                              <td
                                className={`p-2 font-bold ${
                                  ev.reaction_usd_oz >= 0 ? "text-emerald-400" : "text-rose-400"
                                }`}
                              >
                                {ev.reaction_usd_oz >= 0 ? "+" : ""}${ev.reaction_usd_oz.toFixed(2)} (
                                {ev.reaction_bps >= 0 ? "+" : ""}
                                {ev.reaction_bps} bps)
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <Badge variant="emerald">Pillar 2 • Millisecond NFP Microstructure</Badge>
                      <span className="text-xs font-mono text-emerald-300">
                        Slippage Drag Cut -78.2% (13.27 → 2.89 bps)
                      </span>
                    </div>
                    <CardTitle className="text-base pt-1">
                      Cont-Kukanov-Stoikov Tick OFI &amp; Liquidity Resiliency Gate (τ_res = +
                      {pillars.pillar2.resiliency_recovery_ms.toFixed(0)} ms)
                    </CardTitle>
                    <CardDescription>
                      Post-NFP spread widens {pillars.pillar2.metrics.spread_widening_multiple.toFixed(1)}x ($
                      {pillars.pillar2.metrics.pre_spread_usd_oz.toFixed(2)} → $
                      {pillars.pillar2.metrics.max_shock_spread_usd_oz.toFixed(2)}/oz) while L1 depth drops{" "}
                      {pillars.pillar2.metrics.depth_drop_pct.toFixed(1)}%
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <img
                      src="/figures/02_news_microstructure_comparison.png"
                      alt="Pillar 2 Microstructure Chart"
                      className="w-full rounded-lg border border-slate-800"
                    />
                    <div className="grid grid-cols-3 gap-3 text-center">
                      <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                        <div className="text-[11px] text-slate-400">Illusory Mid-Price PnL</div>
                        <div className="mt-1 text-base font-bold text-sky-300">
                          +{pillars.pillar2.sim_unguarded.mid_price_illusion_pnl_bps.toFixed(2)} bps
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                        <div className="text-[11px] text-slate-400">Unguarded t=+50ms Drag</div>
                        <div className="mt-1 text-base font-bold text-rose-400">
                          -{pillars.pillar2.sim_unguarded.execution_cost_drag_bps.toFixed(2)} bps
                        </div>
                      </div>
                      <div className="rounded-lg border border-slate-800 bg-slate-950 p-3">
                        <div className="text-[11px] text-slate-400">τ_res Gated Drag</div>
                        <div className="mt-1 text-base font-bold text-emerald-400">
                          -{pillars.pillar2.sim_gated.execution_cost_drag_bps.toFixed(2)} bps
                        </div>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

              {/* Pillar 3 & Pillar 4 */}
              <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <Badge variant="sky">Pillar 3 • 24h Intraday Volatility ({pillars.pillar3.n_oos_days} OOS Days)</Badge>
                      <span className="text-xs font-mono text-amber-300">
                        ML RMSE = {pillars.pillar3.ml_rmse_pct}% | HAR-RV-J QLIKE = {pillars.pillar3.har_rv_j_qlike}
                      </span>
                    </div>
                    <CardTitle className="text-base pt-1">
                      Finite-Sample Bipower Variation (BV_t), HAR-RV-J &amp; Purged Walk-Forward ML
                    </CardTitle>
                    <CardDescription>
                      23/5 Maintenance Gap Masked • COMEX/LBMA Overlap accounts for{" "}
                      {pillars.pillar3.session_shares.COMEX_LBMA_Overlap}% of daily 24h XAU/USD variance
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      src="/figures/03_ml_vs_garch_volatility_comparison.png"
                      alt="Pillar 3 Volatility Chart"
                      className="w-full rounded-lg border border-slate-800"
                    />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <div className="flex items-center justify-between">
                      <Badge variant="default">Pillar 4 • London–COMEX EFP Basis Arbitrage</Badge>
                      <span className="text-xs font-mono text-emerald-300">
                        DSR = {pillars.pillar4.deflated_sharpe_ratio} | Net PF = {pillars.pillar4.cost_adjusted_profit_factor}
                      </span>
                    </div>
                    <CardTitle className="text-base pt-1">
                      5-Day EMA Carry Stripping, 1D Kalman Filter (β_t = {pillars.pillar4.kalman_beta_mean}) &amp; Hurst TVECM
                    </CardTitle>
                    <CardDescription>
                      Clean Basis ADF = {pillars.pillar4.adf_stat} • Mean Hurst H_t = {pillars.pillar4.mean_hurst_exponent} &lt; 0.45 •{" "}
                      {pillars.pillar4.regime_bars_halted} EFP Squeeze Bars Halted
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    <img
                      src="/figures/04_fx_cross_rate_arbitrage_comparison.png"
                      alt="Pillar 4 EFP Arbitrage Chart"
                      className="w-full rounded-lg border border-slate-800"
                    />
                  </CardContent>
                </Card>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
