const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function apiPost(path: string, body: FormData | Record<string, any>) {
  const isFormData = body instanceof FormData;
  const res = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    body: isFormData ? body : JSON.stringify(body),
    headers: isFormData ? undefined : { "Content-Type": "application/json" },
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Lỗi không xác định từ máy chủ" }));
    throw new Error(err.detail || `Lỗi yêu cầu: ${res.status}`);
  }
  return res.json();
}

export async function apiGet(path: string) {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Lỗi không xác định từ máy chủ" }));
    throw new Error(err.detail || `Lỗi yêu cầu: ${res.status}`);
  }
  return res.json();
}

export async function apiDelete(path: string) {
  const res = await fetch(`${API_BASE}${path}`, { method: "DELETE" });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: "Lỗi không xác định từ máy chủ" }));
    throw new Error(err.detail || `Lỗi yêu cầu: ${res.status}`);
  }
  return res.json();
}

export function createEventSource(path: string): EventSource {
  return new EventSource(`${API_BASE}${path}`);
}

export function formatFileSize(bytes: number): string {
  if (!bytes || bytes === 0) return "";
  if (bytes >= 1073741824) return `${(bytes / 1073741824).toFixed(1)} GB`;
  if (bytes >= 1048576) return `${(bytes / 1048576).toFixed(1)} MB`;
  if (bytes >= 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${bytes} B`;
}

export function formatDuration(seconds: number): string {
  if (!seconds || seconds === 0) return "";
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  if (h > 0) return `${h} giờ ${m} phút`;
  if (m > 0) return `${m} phút ${s} giây`;
  return `${s} giây`;
}

export { API_BASE };
