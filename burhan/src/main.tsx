import React from 'react';
import {createRoot} from 'react-dom/client';
import App from './App';
import './styles.css';
import {Capacitor} from '@capacitor/core';
createRoot(document.getElementById('root')!).render(<React.StrictMode><App/></React.StrictMode>);
if(import.meta.env.PROD&&!Capacitor.isNativePlatform()&&'serviceWorker' in navigator){navigator.serviceWorker.register('/sw.js').catch(error=>console.warn('Offline cache unavailable',error));}
