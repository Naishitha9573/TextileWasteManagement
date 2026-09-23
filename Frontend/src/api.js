const apiBaseUrl = (import.meta.env.VITE_API_URL || '').replace(/\/+$/, '');

export function configureApiBaseUrl() {
  if (!apiBaseUrl) return;

  const nativeFetch = window.fetch.bind(window);
  window.fetch = (input, init) => {
    if (typeof input === 'string' && input.startsWith('/api/')) {
      return nativeFetch(`${apiBaseUrl}${input}`, init);
    }
    return nativeFetch(input, init);
  };
}
