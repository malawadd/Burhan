import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import {readFile} from 'node:fs/promises';

// ONNX Runtime imports its bundled loader dynamically with ?import in dev.
// Serve this single trusted runtime module before Vite's public-import guard.
const bundledOrtLoader={
  name:'burhan-bundled-ort-loader',
  configureServer(server:import('vite').ViteDevServer){
    server.middlewares.use(async(request,response,next)=>{
      if(!['GET','HEAD'].includes(request.method??'')||request.url?.split('?')[0]!=='/ort/ort-wasm-simd-threaded.mjs'){next();return;}
      try{
        const source=await readFile(new URL('./assets/ort/ort-wasm-simd-threaded.mjs',import.meta.url));
        response.statusCode=200;response.setHeader('Content-Type','text/javascript; charset=utf-8');response.setHeader('X-Content-Type-Options','nosniff');response.setHeader('Cache-Control','no-cache');response.end(request.method==='HEAD'?undefined:source);
      }catch(error){next(error);}
    });
  },
};
export default defineConfig({ plugins:[bundledOrtLoader,react()], publicDir:'assets', server:{port:5173,strictPort:true,watch:{ignored:['**/.tools/**','**/android/**','**/delivery/**','**/tests/fixtures/**']}}, build:{target:'es2022',chunkSizeWarningLimit:1200}, worker:{format:'es'} });
