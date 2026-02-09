
import React, { useState } from 'react';
import { Volume2, Share2, Copy, CheckCircle, BrainCircuit, Sparkles } from 'lucide-react';
import { generateSpeech } from '../services/geminiService';
import { playAudio } from '../utils';

interface SolutionDisplayProps {
  content: string;
  isThinking?: boolean;
}

const SolutionDisplay: React.FC<SolutionDisplayProps> = ({ content, isThinking }) => {
  const [isCopied, setIsCopied] = useState(false);
  const [isReading, setIsReading] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(content);
    setIsCopied(true);
    setTimeout(() => setIsCopied(false), 2000);
  };

  const handleShare = async () => {
    if (navigator.share) {
      try {
        await navigator.share({
          title: 'Resolução do Professor Neural',
          text: content,
          url: window.location.href,
        });
      } catch (err) {
        console.error("Erro ao compartilhar:", err);
      }
    } else {
      handleCopy();
      alert("Link copiado para a área de transferência!");
    }
  };

  const handleRead = async () => {
    if (isReading) return;
    setIsReading(true);
    const audioBytes = await generateSpeech(content.slice(0, 800));
    if (audioBytes) {
      await playAudio(audioBytes);
    }
    setIsReading(false);
  };

  return (
    <div className="bg-white dark:bg-slate-900 rounded-5xl shadow-2xl overflow-hidden border border-slate-200 dark:border-slate-800 transition-all duration-500 transform animate-in zoom-in-95 duration-700">
      <div className="bg-brand-600 dark:bg-brand-800 px-8 py-5 flex justify-between items-center text-white">
        <div className="flex items-center gap-4">
          <div className="p-2.5 bg-white/10 rounded-2xl backdrop-blur-xl border border-white/20">
            {isThinking ? <BrainCircuit className="w-5 h-5 animate-pulse" /> : <Sparkles className="w-5 h-5" />}
          </div>
          <div className="flex flex-col">
            <span className="font-extrabold uppercase tracking-[0.2em] text-[9px] opacity-70">Resposta Direta</span>
            <span className="font-bold text-base leading-tight">Resultado Neural</span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button 
            onClick={handleRead}
            disabled={isReading}
            className={`p-2.5 rounded-xl transition-all border border-white/10 ${isReading ? 'bg-white text-brand-600' : 'bg-white/10 hover:bg-white/20 text-white'}`}
          >
            <Volume2 className={`w-4 h-4 ${isReading ? 'animate-pulse' : ''}`} />
          </button>
          <button 
            onClick={handleCopy}
            className={`p-2.5 rounded-xl transition-all border border-white/10 ${isCopied ? 'bg-white text-brand-600' : 'bg-white/10 hover:bg-white/20 text-white'}`}
          >
            {isCopied ? <CheckCircle className="w-4 h-4" /> : <Copy className="w-4 h-4" />}
          </button>
        </div>
      </div>
      
      <div className="p-8 transition-colors">
        <div className="whitespace-pre-wrap text-slate-800 dark:text-slate-200 leading-snug font-semibold text-lg md:text-xl">
          {content}
        </div>
      </div>

      <div className="bg-slate-50 dark:bg-slate-800/50 border-t border-slate-100 dark:border-slate-800 px-8 py-4 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="w-2 h-2 rounded-full bg-green-500"></div>
          <span className="text-[10px] font-black text-slate-400 dark:text-slate-500 uppercase tracking-widest">Otimizado para Clareza</span>
        </div>
        <button onClick={handleShare} className="group flex items-center gap-2">
            <span className="text-[10px] font-black text-slate-400 group-hover:text-brand-500 uppercase tracking-widest transition-colors">Compartilhar</span>
            <Share2 className="w-4 h-4 text-slate-400 group-hover:text-brand-500 transition-colors" />
        </button>
      </div>
    </div>
  );
};

export default SolutionDisplay;
