import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins:[react()], publicDir:'assets', server:{port:5173,strictPort:true,watch:{ignored:['**/.tools/**','**/android/**','**/delivery/**','**/tests/fixtures/**']}}, build:{target:'es2022',chunkSizeWarningLimit:1200}, worker:{format:'es'} });
