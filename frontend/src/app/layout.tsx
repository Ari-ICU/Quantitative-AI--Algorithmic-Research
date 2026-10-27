import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "XAU/USD Quantitative AI & macOS MT5 Bridge Terminal",
  description:
    "Institutional Spot Gold (XAU/USD) Quantitative Research & Live macOS MetaTrader Bridge powered by Next.js, Tailwind CSS, Framer Motion, shadcn/ui, and Python.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-[#060b16] text-slate-100 antialiased selection:bg-amber-400 selection:text-slate-950">
        {children}
      </body>
    </html>
  );
}
