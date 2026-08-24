import Image from "next/image";
import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 text-white selection:bg-indigo-500/30">
      {/* Navbar */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-slate-950/50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-lg">
              S
            </div>
            <span className="font-semibold text-lg tracking-tight">Sakai Studio</span>
          </div>
          <div className="flex items-center gap-6 text-sm font-medium">
            <Link href="#features" className="text-slate-400 hover:text-white transition-colors">Features</Link>
            <Link href="#pricing" className="text-slate-400 hover:text-white transition-colors">Pricing</Link>
            <Link href="/login" className="text-slate-400 hover:text-white transition-colors">Sign In</Link>
            <Link href="/dashboard" className="bg-white text-slate-900 px-4 py-2 rounded-full hover:bg-slate-200 transition-colors">
              Get Started
            </Link>
          </div>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        <div className="absolute inset-0 bg-[url('/grid.svg')] bg-center [mask-image:linear-gradient(180deg,white,rgba(255,255,255,0))]" />
        
        <div className="max-w-7xl mx-auto px-6 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm font-medium mb-8">
            <span className="w-2 h-2 rounded-full bg-indigo-500 animate-pulse" />
            SAM 2 + ProPainter Integration Live
          </div>
          
          <h1 className="text-5xl lg:text-7xl font-bold tracking-tight mb-8">
            Flawless Video AI <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
              In Your Browser.
            </span>
          </h1>
          
          <p className="text-lg lg:text-xl text-slate-400 max-w-2xl mx-auto mb-10">
            Remove logos, hardcoded subtitles, and unwanted objects from your videos with a single click. Powered by state-of-the-art AI tracking and inpainting.
          </p>
          
          <div className="flex items-center justify-center gap-4">
            <Link href="/dashboard" className="px-8 py-4 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium transition-all hover:scale-105 active:scale-95 shadow-[0_0_40px_-10px_rgba(79,70,229,0.5)]">
              Start Processing Now
            </Link>
            <Link href="#demo" className="px-8 py-4 rounded-full bg-white/5 hover:bg-white/10 text-white font-medium transition-colors border border-white/10">
              View Demo
            </Link>
          </div>
        </div>
      </section>

      {/* Interactive Demo/Features Section */}
      <section id="demo" className="py-24 bg-slate-900/50 border-y border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col lg:flex-row gap-16 items-center">
            <div className="flex-1 space-y-8">
              <h2 className="text-3xl lg:text-4xl font-bold">Inpaint like magic. <br/>Track like radar.</h2>
              <p className="text-slate-400 text-lg">
                Draw a box around the object you want gone. Our SAM 2 engine precisely tracks its boundaries across frames, while LaMa reconstructs the background flawlessly.
              </p>
              
              <ul className="space-y-4">
                {[
                  "Bi-directional Tracking for zero drift",
                  "Hardware accelerated rendering",
                  "Auto OCR layout preservation"
                ].map((feature, idx) => (
                  <li key={idx} className="flex items-center gap-3 text-slate-300">
                    <svg className="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7"></path></svg>
                    {feature}
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="flex-1 w-full relative">
              <div className="aspect-video rounded-2xl bg-slate-800 border border-white/10 overflow-hidden shadow-2xl relative group">
                <div className="absolute inset-0 flex items-center justify-center bg-black/40 group-hover:bg-black/20 transition-colors cursor-pointer">
                  <div className="w-16 h-16 rounded-full bg-white/10 backdrop-blur-md flex items-center justify-center border border-white/20 hover:scale-110 transition-transform">
                    <div className="w-0 h-0 border-y-8 border-y-transparent border-l-[12px] border-l-white ml-1" />
                  </div>
                </div>
                {/* Simulated UI Overlay */}
                <div className="absolute bottom-4 left-4 right-4 flex items-center gap-4 bg-slate-950/80 backdrop-blur border border-white/10 rounded-xl p-3">
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div className="w-2/3 h-full bg-indigo-500 rounded-full" />
                  </div>
                  <span className="text-xs font-mono text-slate-400">01:24</span>
                </div>
                <div className="absolute top-1/3 left-1/4 w-32 h-24 border-2 border-dashed border-indigo-400 rounded-lg bg-indigo-400/10 flex items-center justify-center">
                  <span className="text-xs font-mono text-indigo-300 bg-slate-900/80 px-2 py-1 rounded">SAM2 Tracking</span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/5 py-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-xs">S</div>
            <span className="font-medium text-slate-300">Sakai Studio SaaS</span>
          </div>
          <p className="text-slate-500 text-sm">© 2026 Sakai Studio. All rights reserved.</p>
        </div>
      </footer>
    </main>
  );
}
