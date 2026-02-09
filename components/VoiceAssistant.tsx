
import React, { useState, useEffect, useRef } from 'react';
import { Mic, MicOff, Headset, Loader2, Sparkles } from 'lucide-react';
import { GoogleGenAI, LiveServerMessage, Modality, Blob } from '@google/genai';
import { decode, encode, decodeAudioData } from '../utils';

const VoiceAssistant: React.FC = () => {
  const [isActive, setIsActive] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [transcriptions, setTranscriptions] = useState<{ role: string, text: string }[]>([]);
  
  const nextStartTimeRef = useRef(0);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const sessionRef = useRef<any>(null);
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set());

  const startSession = async () => {
    setIsConnecting(true);
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY || '' });
    
    try {
      const inputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      outputAudioContextRef.current = outputCtx;

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      const sessionPromise = ai.live.connect({
        model: 'gemini-2.5-flash-native-audio-preview-12-2025',
        callbacks: {
          onopen: () => {
            const source = inputCtx.createMediaStreamSource(stream);
            const scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
            
            scriptProcessor.onaudioprocess = (e) => {
              const inputData = e.inputBuffer.getChannelData(0);
              const l = inputData.length;
              const int16 = new Int16Array(l);
              for (let i = 0; i < l; i++) {
                int16[i] = inputData[i] * 32768;
              }
              const pcmBlob: Blob = {
                data: encode(new Uint8Array(int16.buffer)),
                mimeType: 'audio/pcm;rate=16000',
              };
              
              sessionPromise.then((session) => {
                session.sendRealtimeInput({ media: pcmBlob });
              });
            };

            source.connect(scriptProcessor);
            scriptProcessor.connect(inputCtx.destination);
            setIsConnecting(false);
            setIsActive(true);
          },
          onmessage: async (message: LiveServerMessage) => {
            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              setTranscriptions(prev => [...prev, { role: 'professor', text }]);
            }

            const base64Audio = message.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
            if (base64Audio && outputAudioContextRef.current) {
              const ctx = outputAudioContextRef.current;
              nextStartTimeRef.current = Math.max(nextStartTimeRef.current, ctx.currentTime);
              
              const buffer = await decodeAudioData(decode(base64Audio), ctx, 24000, 1);
              const source = ctx.createBufferSource();
              source.buffer = buffer;
              source.connect(ctx.destination);
              source.start(nextStartTimeRef.current);
              nextStartTimeRef.current += buffer.duration;
              sourcesRef.current.add(source);
            }

            if (message.serverContent?.interrupted) {
              sourcesRef.current.forEach(s => s.stop());
              sourcesRef.current.clear();
            }
          },
          onerror: (e) => console.error("Live API Error:", e),
          onclose: () => {
            setIsActive(false);
            setIsConnecting(false);
          }
        },
        config: {
          responseModalities: [Modality.AUDIO],
          systemInstruction: 'Você é o Professor de Matemática. Converse com o aluno sobre dúvidas de matemática, física e química em tempo real.',
          outputAudioTranscription: {},
        }
      });

      sessionRef.current = await sessionPromise;
    } catch (err) {
      console.error(err);
      setIsConnecting(false);
    }
  };

  const stopSession = () => {
    sessionRef.current?.close();
    setIsActive(false);
  };

  return (
    <div className="flex flex-col items-center gap-8 p-10 bg-white dark:bg-slate-900 rounded-[2.5rem] border border-slate-200 dark:border-slate-800 shadow-2xl transition-all">
      <div className="flex flex-col items-center gap-2">
        <div className="p-4 bg-brand-50 dark:bg-brand-900/30 rounded-2xl text-brand-600 dark:text-brand-400 mb-2">
          <Headset className="w-10 h-10" />
        </div>
        <h3 className="text-2xl font-black text-slate-900 dark:text-white">Laboratório de Voz</h3>
        <p className="text-slate-500 dark:text-slate-400 text-center max-w-xs font-medium text-sm">
          Fale com o Professor sobre fórmulas complexas e teoria científica.
        </p>
      </div>

      <div className="relative">
        {isActive && (
          <div className="absolute -inset-8 bg-brand-500 rounded-full animate-ping opacity-10"></div>
        )}
        <button
          onClick={isActive ? stopSession : startSession}
          disabled={isConnecting}
          className={`relative z-10 w-28 h-28 rounded-full flex items-center justify-center transition-all transform hover:scale-110 active:scale-95 shadow-2xl ${
            isActive ? 'bg-red-500 text-white shadow-red-200 dark:shadow-none' : 'bg-brand-600 text-white shadow-brand-200 dark:shadow-none'
          } ${isConnecting ? 'opacity-50 cursor-wait' : ''}`}
        >
          {isConnecting ? (
            <Loader2 className="w-12 h-12 animate-spin" />
          ) : isActive ? (
            <MicOff className="w-12 h-12" />
          ) : (
            <Mic className="w-12 h-12" />
          )}
        </button>
      </div>

      <div className="w-full h-32 overflow-y-auto space-y-3 px-4 bg-slate-50 dark:bg-slate-800/50 rounded-2xl p-4 border border-slate-100 dark:border-slate-800">
        {transcriptions.length === 0 ? (
          <div className="h-full flex items-center justify-center text-slate-400 text-xs uppercase tracking-widest font-bold">
            Aguardando sinal...
          </div>
        ) : (
          transcriptions.slice(-10).map((t, i) => (
            <div key={i} className={`flex gap-2 text-sm ${t.role === 'professor' ? 'text-brand-600 dark:text-brand-400 font-bold' : 'text-slate-500 dark:text-slate-400 font-medium'}`}>
              <span className="shrink-0 uppercase text-[10px] opacity-60 mt-0.5">{t.role === 'professor' ? 'Prof:' : 'Você:'}</span>
              <span className="leading-tight">{t.text}</span>
            </div>
          ))
        )}
      </div>

      {isActive && (
        <div className="flex items-end gap-1.5 h-10">
          {[0.4, 0.7, 1, 0.6, 0.8, 0.5, 0.9, 0.3].map((h, i) => (
            <div 
              key={i} 
              className="w-2 bg-brand-500 dark:bg-brand-400 rounded-full animate-pulse transition-all duration-300" 
              style={{ height: `${h * 100}%`, animationDelay: `${i * 0.15}s` }}
            ></div>
          ))}
        </div>
      )}
    </div>
  );
};

export default VoiceAssistant;
