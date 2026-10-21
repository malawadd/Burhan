const CACHE='burhan-offline-v1';
self.addEventListener('install',e=>e.waitUntil(caches.open(CACHE).then(c=>c.addAll(['/','/catalogue.json','/catalogue-ar.json','/models/manifest.json','/icon.svg','/fonts/NotoSansArabic-Regular.ttf','/fonts/NotoSans-Regular.ttf']))));
self.addEventListener('activate',e=>e.waitUntil(self.clients.claim()));
self.addEventListener('fetch',e=>{if(e.request.method!=='GET'||new URL(e.request.url).origin!==self.location.origin)return;e.respondWith((async()=>{const cache=await caches.open(CACHE);const url=new URL(e.request.url);const hit=await cache.match(e.request);if(hit&&/\.(onnx|wasm|ttf)$/.test(url.pathname))return hit;try{const r=await fetch(e.request);if(r.ok)await cache.put(e.request,r.clone());return r;}catch(error){if(hit)return hit;throw error;}})());});
