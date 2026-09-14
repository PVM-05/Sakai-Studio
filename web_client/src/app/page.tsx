import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen bg-slate-950 text-white selection:bg-indigo-500/30">
      {/* Thanh điều hướng */}
      <nav className="fixed top-0 w-full z-50 border-b border-white/5 bg-slate-950/50 backdrop-blur-md">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-lg text-white shadow-md">
              S
            </div>
            <span className="font-semibold text-lg tracking-tight">Sakai Studio</span>
          </div>
          <div className="flex items-center gap-6 text-sm font-medium">
            <Link href="#features" className="text-slate-400 hover:text-white transition-colors">
              Tính năng
            </Link>
            <Link href="#pricing" className="text-slate-400 hover:text-white transition-colors">
              Bảng giá
            </Link>
            <Link href="/dashboard" className="text-slate-400 hover:text-white transition-colors">
              Đăng nhập
            </Link>
            <Link
              href="/dashboard"
              className="bg-white text-slate-900 px-5 py-2 rounded-full hover:bg-slate-200 transition-colors font-semibold"
            >
              Bắt đầu ngay
            </Link>
          </div>
        </div>
      </nav>

      {/* Phần giới thiệu chính */}
      <section className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
        <div className="max-w-7xl mx-auto px-6 relative z-10 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-300 text-sm font-medium mb-8">
            <span className="w-2 h-2 rounded-full bg-indigo-500" />
            Tích hợp mô hình SAM 2 và ProPainter chất lượng cao
          </div>

          <h1 className="text-5xl lg:text-7xl font-bold tracking-tight mb-8">
            Xóa chữ và phụ đề video <br />
            <span className="text-transparent bg-clip-text bg-gradient-to-r from-indigo-400 via-purple-400 to-pink-400">
              Trực tiếp trên trình duyệt
            </span>
          </h1>

          <p className="text-lg lg:text-xl text-slate-400 max-w-2xl mx-auto mb-10 leading-relaxed">
            Loại bỏ hình mờ, phụ đề gắn cứng và các chi tiết thừa trong video nhanh chóng. Ứng dụng công nghệ bám đuổi chuyển động và khôi phục nền tiên tiến nhất.
          </p>

          <div className="flex items-center justify-center gap-4">
            <Link
              href="/dashboard"
              className="px-8 py-4 rounded-full bg-indigo-600 hover:bg-indigo-700 text-white font-medium transition-all hover:scale-105 active:scale-95 shadow-[0_0_40px_-10px_rgba(79,70,229,0.5)]"
            >
              Trải nghiệm ngay
            </Link>
            <Link
              href="#demo"
              className="px-8 py-4 rounded-full bg-white/5 hover:bg-white/10 text-white font-medium transition-colors border border-white/10"
            >
              Xem mô phỏng
            </Link>
          </div>
        </div>
      </section>

      {/* Phần tính năng tương tác */}
      <section id="demo" className="py-24 bg-slate-900/50 border-y border-white/5">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex flex-col lg:flex-row gap-16 items-center">
            <div className="flex-1 space-y-8">
              <h2 className="text-3xl lg:text-4xl font-bold">
                Khôi phục nền mượt mà. <br />
                Bám đuổi chuyển động chính xác.
              </h2>
              <p className="text-slate-400 text-lg leading-relaxed">
                Đánh dấu vùng đối tượng cần xóa trên video. Hệ thống tự động nhận diện đường viền qua từng khung hình và tái tạo bề mặt nền video tự nhiên liền mạch.
              </p>

              <ul className="space-y-4">
                {[
                  "Bám đuổi hai chiều hạn chế tối đa độ lệch vùng",
                  "Tăng tốc xử lý phần cứng tối ưu hiệu năng",
                  "Tự động nhận diện chữ bảo toàn bố cục khung hình",
                ].map((feature, idx) => (
                  <li key={idx} className="flex items-center gap-3 text-slate-300">
                    <svg className="w-5 h-5 text-indigo-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M5 13l4 4L19 7" />
                    </svg>
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

                {/* Mô phỏng khung điều khiển */}
                <div className="absolute bottom-4 left-4 right-4 flex items-center gap-4 bg-slate-950/80 backdrop-blur border border-white/10 rounded-xl p-3">
                  <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
                    <div className="w-2/3 h-full bg-indigo-500 rounded-full" />
                  </div>
                  <span className="text-xs font-mono text-slate-400">01:24</span>
                </div>
                <div className="absolute top-1/3 left-1/4 w-36 h-20 border-2 border-dashed border-indigo-400 rounded-lg bg-indigo-400/10 flex items-center justify-center">
                  <span className="text-xs font-mono text-indigo-300 bg-slate-900/80 px-2 py-1 rounded border border-indigo-500/20">
                    Bám đuổi SAM 2
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Chân trang */}
      <footer className="border-t border-white/5 py-12">
        <div className="max-w-7xl mx-auto px-6 flex flex-col md:flex-row justify-between items-center gap-6">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center font-bold text-xs text-white">
              S
            </div>
            <span className="font-medium text-slate-300">Sakai Studio</span>
          </div>
          <p className="text-slate-500 text-sm">© 2026 Sakai Studio. Bảo lưu mọi quyền.</p>
        </div>
      </footer>
    </main>
  );
}
