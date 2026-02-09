
import React, { useRef, useState, useEffect, useCallback } from 'react';
import { RefreshCw, Zap, AlertCircle, Maximize2, ShieldCheck, Camera as CameraIcon, Power, Plus, Minus } from 'lucide-react';

interface CameraCaptureProps {
  onCapture: (base64Image: string) => void;
  isProcessing?: boolean;
}

const CameraCapture: React.FC<CameraCaptureProps> = ({ onCapture, isProcessing }) => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Zoom States
  const [zoom, setZoom] = useState(1);
  const [zoomCapabilities, setZoomCapabilities] = useState<{ min: number, max: number, step: number } | null>(null);
  const touchStartDist = useRef<number | null>(null);

  const stopCamera = useCallback(() => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsCameraActive(false);
    setZoomCapabilities(null);
    setZoom(1);
  }, [stream]);

  const startCamera = useCallback(async () => {
    setError(null);
    try {
      const constraints = {
        video: { 
          facingMode: 'environment',
          width: { ideal: 1920 },
          height: { ideal: 1080 }
        },
        audio: false
      };
      
      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      const videoTrack = mediaStream.getVideoTracks()[0];
      
      // Check for zoom capabilities (Chrome/Android support)
      const capabilities = (videoTrack as any).getCapabilities?.() || {};
      if (capabilities.zoom) {
        setZoomCapabilities({
          min: capabilities.zoom.min,
          max: capabilities.zoom.max,
          step: capabilities.zoom.step
        });
        setZoom(capabilities.zoom.min);
      }

      setStream(mediaStream);
      setIsCameraActive(true);
    } catch (err: any) {
      console.error("Erro ao acessar câmera:", err);
      let msg = "Não foi possível acessar a câmera.";
      if (err.name === 'NotAllowedError') msg = "Acesso à câmera negado. Verifique as permissões do navegador.";
      if (err.name === 'NotFoundError') msg = "Nenhuma câmera encontrada no dispositivo.";
      setError(msg);
      setIsCameraActive(false);
    }
  }, []);

  useEffect(() => {
    if (isCameraActive && stream && videoRef.current) {
      videoRef.current.srcObject = stream;
      videoRef.current.play().catch(e => console.error("Erro ao dar play no vídeo:", e));
    }
  }, [isCameraActive, stream]);

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
    };
  }, [stream]);

  const applyZoom = async (value: number) => {
    if (!stream || !zoomCapabilities) return;
    const track = stream.getVideoTracks()[0];
    try {
      await (track as any).applyConstraints({
        advanced: [{ zoom: value }]
      });
      setZoom(value);
    } catch (e) {
      console.warn("Hardware zoom not supported or failed:", e);
    }
  };

  // Pinch-to-zoom handlers
  const handleTouchStart = (e: React.TouchEvent) => {
    if (e.touches.length === 2) {
      touchStartDist.current = Math.hypot(
        e.touches[0].pageX - e.touches[1].pageX,
        e.touches[0].pageY - e.touches[1].pageY
      );
    }
  };

  const handleTouchMove = (e: React.TouchEvent) => {
    if (e.touches.length === 2 && touchStartDist.current && zoomCapabilities) {
      const dist = Math.hypot(
        e.touches[0].pageX - e.touches[1].pageX,
        e.touches[0].pageY - e.touches[1].pageY
      );
      const delta = (dist - touchStartDist.current) / 100;
      const nextZoom = Math.min(Math.max(zoom + delta, zoomCapabilities.min), zoomCapabilities.max);
      applyZoom(nextZoom);
      touchStartDist.current = dist;
    }
  };

  const capturePhoto = () => {
    if (videoRef.current && canvasRef.current && !isProcessing && isCameraActive) {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      
      const vw = video.videoWidth;
      const vh = video.videoHeight;

      const cropWidth = vw * 0.8;
      const cropHeight = vh * 0.5;
      const startX = (vw - cropWidth) / 2;
      const startY = (vh - cropHeight) / 2;

      canvas.width = cropWidth;
      canvas.height = cropHeight;
      
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.drawImage(
          video, 
          startX, startY, cropWidth, cropHeight,
          0, 0, cropWidth, cropHeight
        );
        
        const dataUrl = canvas.toDataURL('image/jpeg', 0.9);
        onCapture(dataUrl.split(',')[1]);
      }
    }
  };

  if (!isCameraActive && !error) {
    return (
      <div className="relative w-full aspect-[3/4] md:aspect-[16/10] bg-slate-100 dark:bg-slate-900 rounded-[2.5rem] overflow-hidden shadow-2xl border-4 border-white dark:border-slate-800 flex flex-col items-center justify-center transition-all duration-500 group">
        <div className="absolute inset-0 bg-gradient-to-br from-brand-500/10 to-transparent opacity-50"></div>
        <div className="relative z-10 flex flex-col items-center gap-6 p-8 text-center">
          <div className="w-24 h-24 bg-white dark:bg-slate-800 rounded-full flex items-center justify-center shadow-2xl mb-2 group-hover:scale-110 transition-transform duration-500 border border-slate-100 dark:border-slate-700">
            <CameraIcon className="w-10 h-10 text-brand-600 dark:text-brand-400" />
          </div>
          <div className="space-y-3">
            <h3 className="text-3xl font-black text-slate-900 dark:text-white tracking-tight">Câmera em Espera</h3>
            <p className="text-slate-500 dark:text-slate-400 font-medium max-w-xs mx-auto">
              Aponte para o cálculo e ative a câmera para resolver.
            </p>
          </div>
          <button 
            onClick={startCamera}
            className="flex items-center gap-4 px-10 py-5 bg-brand-600 hover:bg-brand-700 text-white rounded-3xl font-black text-sm uppercase tracking-widest shadow-2xl shadow-brand-600/30 transition-all hover:scale-105 active:scale-95"
          >
            <Power className="w-6 h-6" />
            Ligar Câmera
          </button>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex flex-col items-center justify-center p-12 text-center bg-white dark:bg-slate-900 rounded-5xl border-2 border-slate-100 dark:border-slate-800 shadow-xl min-h-[400px]">
        <div className="bg-red-50 dark:bg-red-900/20 p-6 rounded-full mb-6">
          <AlertCircle className="w-16 h-16 text-red-500" />
        </div>
        <h3 className="text-2xl font-black mb-3 text-slate-900 dark:text-white">Acesso Necessário</h3>
        <p className="text-slate-500 dark:text-slate-400 mb-8 max-w-xs font-medium">{error}</p>
        <button 
          onClick={startCamera}
          className="px-8 py-4 bg-brand-600 text-white rounded-2xl hover:bg-brand-700 transition-all font-bold shadow-lg shadow-brand-600/20"
        >
          Tentar Novamente
        </button>
      </div>
    );
  }

  return (
    <div 
      className="relative w-full aspect-[3/4] md:aspect-[16/10] bg-slate-900 rounded-[2.5rem] overflow-hidden shadow-2xl border-4 border-white dark:border-slate-800 transition-all duration-500"
      onTouchStart={handleTouchStart}
      onTouchMove={handleTouchMove}
    >
      <video 
        ref={videoRef} 
        autoPlay 
        playsInline 
        muted
        className={`w-full h-full object-cover transition-opacity duration-700 ${isProcessing ? 'opacity-40 grayscale' : 'opacity-100'}`}
      />
      
      <canvas ref={canvasRef} className="hidden" />

      {!isProcessing && isCameraActive && <div className="scan-line-focused" />}

      {/* Focus Area Dimming */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="w-full h-1/4 bg-black/40 backdrop-blur-[2px]"></div>
        <div className="flex w-full h-1/2">
          <div className="w-[10%] bg-black/40 backdrop-blur-[2px]"></div>
          <div className="w-4/5"></div>
          <div className="w-[10%] bg-black/40 backdrop-blur-[2px]"></div>
        </div>
        <div className="w-full h-1/4 bg-black/40 backdrop-blur-[2px]"></div>
      </div>

      {/* Zoom Slider UI */}
      {zoomCapabilities && !isProcessing && (
        <div className="absolute right-6 top-1/2 -translate-y-1/2 flex flex-col items-center gap-4 bg-black/40 backdrop-blur-xl p-3 rounded-full border border-white/10 z-30">
          <button onClick={() => applyZoom(Math.min(zoom + 0.5, zoomCapabilities.max))} className="p-1 text-white hover:text-brand-400"><Plus className="w-5 h-5"/></button>
          <div className="h-32 w-1.5 bg-white/20 rounded-full relative overflow-hidden">
            <div 
              className="absolute bottom-0 left-0 right-0 bg-brand-500 transition-all duration-200" 
              style={{ height: `${((zoom - zoomCapabilities.min) / (zoomCapabilities.max - zoomCapabilities.min)) * 100}%` }}
            />
          </div>
          <button onClick={() => applyZoom(Math.max(zoom - 0.5, zoomCapabilities.min))} className="p-1 text-white hover:text-brand-400"><Minus className="w-5 h-5"/></button>
          <div className="text-[10px] text-white font-black">{zoom.toFixed(1)}x</div>
        </div>
      )}

      {/* Interface Overlays */}
      <div className="absolute inset-0 flex flex-col justify-between p-8 pointer-events-none">
        <div className="flex justify-between items-start pointer-events-auto">
          <div className="bg-white/10 backdrop-blur-2xl px-5 py-2.5 rounded-2xl text-white text-[11px] font-black tracking-widest flex items-center gap-2 border border-white/20">
            <ShieldCheck className="w-4 h-4 text-brand-400" />
            FOCO ATIVO
          </div>
          <button 
            onClick={stopCamera}
            className="p-3 bg-red-500/40 hover:bg-red-500/60 backdrop-blur-2xl rounded-2xl text-white border border-white/20 transition-all active:scale-90"
            title="Desligar Câmera"
          >
            <Power className="w-6 h-6" />
          </button>
        </div>

        <div className="flex flex-col items-center gap-8 pointer-events-auto">
          <div className="bg-black/60 backdrop-blur-md px-8 py-3 rounded-full text-white/90 text-sm font-bold border border-white/10 shadow-2xl">
            {zoom > 1 ? `Zoom: ${zoom.toFixed(1)}x` : 'Enquadre o problema no centro'}
          </div>
          
          <div className="flex items-center gap-12">
            <button 
              onClick={startCamera}
              className="p-5 bg-white/10 hover:bg-white/20 backdrop-blur-3xl rounded-3xl text-white transition-all transform hover:rotate-180 border border-white/20 active:scale-90"
            >
              <RefreshCw className="w-7 h-7" />
            </button>
            
            <button 
              onClick={capturePhoto}
              disabled={isProcessing}
              className={`w-24 h-24 rounded-full flex items-center justify-center border-[8px] border-brand-500/20 transition-all transform hover:scale-105 active:scale-90 shadow-[0_0_40px_rgba(37,99,235,0.4)] ${isProcessing ? 'opacity-50 cursor-not-allowed' : 'bg-white hover:bg-brand-50'}`}
            >
              <div className={`w-16 h-16 rounded-full border-[5px] border-brand-600 flex items-center justify-center ${isProcessing ? 'animate-spin border-t-transparent' : ''}`}>
                 {!isProcessing && <div className="w-10 h-10 bg-brand-600 rounded-full" />}
              </div>
            </button>

            <button className="p-5 bg-white/10 backdrop-blur-3xl rounded-3xl text-white border border-white/20 opacity-50 cursor-not-allowed">
              <Maximize2 className="w-7 h-7" />
            </button>
          </div>
        </div>
      </div>

      {/* Guide Markers */}
      <div className="absolute inset-0 pointer-events-none flex items-center justify-center">
        <div className="w-4/5 h-1/2 border-2 border-white/20 rounded-3xl relative shadow-[0_0_0_1000px_rgba(0,0,0,0.1)]">
          <div className="absolute -top-1 -left-1 w-12 h-12 border-t-8 border-l-8 border-brand-500 rounded-tl-3xl shadow-[-5px_-5px_10px_rgba(37,99,235,0.3)]" />
          <div className="absolute -top-1 -right-1 w-12 h-12 border-t-8 border-r-8 border-brand-500 rounded-tr-3xl shadow-[5px_-5px_10px_rgba(37,99,235,0.3)]" />
          <div className="absolute -bottom-1 -left-1 w-12 h-12 border-b-8 border-l-8 border-brand-500 rounded-bl-3xl shadow-[-5px_5px_10px_rgba(37,99,235,0.3)]" />
          <div className="absolute -bottom-1 -right-1 w-12 h-12 border-b-8 border-r-8 border-brand-500 rounded-br-3xl shadow-[5px_5px_10px_rgba(37,99,235,0.3)]" />
        </div>
      </div>
    </div>
  );
};

export default CameraCapture;
