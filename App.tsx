
import React, { useState, useEffect } from 'react';
import { 
  Camera as CameraIcon, 
  MessageSquare, 
  History as HistoryIcon, 
  Mic, 
  Sparkles,
  Loader2,
  ChevronRight,
  Sun,
  Moon,
  ArrowRight,
  Cpu,
  Zap,
  Trash2,
  DollarSign
} from 'lucide-react';
import Logo from './components/Logo';
import CameraCapture from './components/CameraCapture';
import SolutionDisplay from './components/SolutionDisplay';
import VoiceAssistant from './components/VoiceAssistant';
import { solveFromImage, chatWithProfessor } from './services/geminiService';
import { AppMode, Solution, ResponseMode } from './types';

// Placeholder para o AdMob (Para ser usado após npx cap add android/ios)
// import { AdMob, BannerAdSize, BannerAdPosition, BannerAdPluginEvents, AdMobBannerSize } from 'capacitor-admob';

const AdPlaceholder = () => (
  <div className="w-full p-4 bg-slate-100 dark:bg-slate-900/50 border-2 border-dashed border-slate-200 dark:border-slate-800 rounded-3xl flex flex-col items-center justify-center gap-2 min-h-[100px] my-4 opacity-60">
    <div className="flex items-center gap-2 text-slate-400">
      <DollarSign className="w-4 h-4" />
      <span className="text-[10px] font-black uppercase tracking-widest">Espaço para Anúncio AdMob</span>
    </div>
  </div>
);

const App: React.FC = () => {
  const [mode, setMode] = useState<AppMode>(AppMode.CAMERA);
  const [isSolving, setIsSolving] = useState(false);
  const [activeSolutions, setActiveSolutions] = useState<Solution[]>([]);
  const [history, setHistory] = useState<Solution[]>([]);
  const [chatInput, setChatInput] = useState('');
  const [isThinkingMode, setIsThinkingMode] = useState(true);
  const [responseMode, setResponseMode] = useState<ResponseMode>(ResponseMode.EXPLAINED);
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('theme') as 'light' | 'dark' || 
             (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    }
    return 'light';
  });

  // Inicialização do AdMob quando o componente monta (apenas em ambiente nativo)
  useEffect(() => {
    /* 
    const initAdMob = async () => {
      try {
        await AdMob.initialize({
          requestTrackingAuthorization: true,
          testingDevices: ['YOUR_DEVICE_ID'],
          initializeForTesting: true,
        });
      } catch (e) {
        console.log("Ambiente Web detectado, AdMob Nativo ignorado.");
      }
    };
    initAdMob();
    */
  }, []);

  useEffect(() => {
    const root = window.document.documentElement;
    if (theme === 'dark') root.classList.add('dark');
    else root.classList.remove('dark');
    localStorage.setItem('theme', theme);
  }, [theme]);

  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');

  const handleCapture = async (base64Image: string) => {
    if (!process.env.API_KEY) {
        alert("Configuração Necessária: Defina a variável de ambiente API_KEY no seu ambiente de build.");
        return;
    }
    setIsSolving(true);
    try {
      const result = await solveFromImage(base64Image, isThinkingMode, responseMode);
      
      const newEntry: Solution = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        answer: result,
        image: base64Image,
        type: 'math'
      };
      
      setActiveSolutions(prev => [newEntry, ...prev]);
      setHistory(prev => [newEntry, ...prev]);
      
      if (window.navigator && window.navigator.vibrate) window.navigator.vibrate(50);
    } catch (err) {
      console.error(err);
      alert("Falha na conexão neural.");
    } finally {
      setIsSolving(false);
    }
  };

  const handleChat = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatInput.trim()) return;
    
    setIsSolving(true);
    const text = chatInput;
    setChatInput('');
    try {
      const result = await chatWithProfessor(text, responseMode);
      const newEntry: Solution = {
        id: Date.now().toString(),
        timestamp: Date.now(),
        answer: result,
        type: 'math'
      };
      setActiveSolutions(prev => [newEntry, ...prev]);
      setHistory(prev => [newEntry, ...prev]);
    } catch (err) {
      console.error(err);
    } finally {
      setIsSolving(false);
    }
  };

  const clearSession = () => {
    setActiveSolutions([]);
  };

  return (
    <div className="min-h-screen bg-slate-50 dark:bg-slate-950 flex flex-col transition-all duration-700 select-none">
      {/* Safe Area Inset para Apps Nativo */}
      <div className="h-[env(safe-area-inset-top)] bg-white dark:bg-slate-900"></div>
      
      <div className="fixed inset-0 pointer-events-none overflow-hidden opacity-30 dark:opacity-20 z-0">
        <div className="absolute top-[-10%] left-[-10%] w-[50%] h-[50%] bg-blue-500/20 blur-[140px] rounded-full animate-pulse"></div>
        <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-brand-500/20 blur-[140px] rounded-full animate-pulse"></div>
      </div>

      <header className="sticky top-0 z-[60] glass border-b border-slate-200/50 dark:border-slate-800/50">
        <div className="max-w-5xl mx-auto px-4 py-5 flex items-center justify-between">
          <div className="flex items-center gap-3 sm:gap-5">
            <div className="relative">
              <Logo className="w-10 h-10 sm:w-12 sm:h-12 shadow-2xl shadow-brand-600/20" />
              <div className="absolute -bottom-1 -right-1 w-4 h-4 sm:w-5 sm:h-5 bg-green-500 border-2 border-white dark:border-slate-900 rounded-full"></div>
            </div>
            <div className="hidden sm:block">
              <h1 className="text-xl font-black text-slate-900 dark:text-white leading-none tracking-tight">Professor Neural</h1>
              <p className="text-[10px] text-brand-600 dark:text-brand-400 font-extrabold tracking-[0.2em] uppercase mt-1.5 flex items-center gap-1.5">
                <Cpu className="w-3 h-3" />
                Mobile Neural Engine
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-2 sm:gap-3">
            <div className="flex bg-slate-100 dark:bg-slate-800 p-1 rounded-2xl border border-slate-200 dark:border-slate-700">
               <button onClick={() => setResponseMode(ResponseMode.SIMPLE)} className={`px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${responseMode === ResponseMode.SIMPLE ? 'bg-white dark:bg-slate-700 text-brand-600 dark:text-brand-400 shadow-sm' : 'text-slate-400'}`}>Simples</button>
               <button onClick={() => setResponseMode(ResponseMode.EXPLAINED)} className={`px-3 py-1.5 rounded-xl text-[10px] font-black uppercase tracking-widest transition-all ${responseMode === ResponseMode.EXPLAINED ? 'bg-white dark:bg-slate-700 text-brand-600 dark:text-brand-400 shadow-sm' : 'text-slate-400'}`}>Explicação</button>
            </div>
            <button onClick={() => setIsThinkingMode(!isThinkingMode)} className={`p-2.5 sm:px-4 sm:py-2.5 rounded-2xl flex items-center gap-2.5 transition-all font-black text-[11px] border ${isThinkingMode ? 'bg-brand-600 text-white border-brand-500 shadow-xl shadow-brand-600/30' : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400 border-transparent'}`}>
              <Zap className={`w-4 h-4 ${isThinkingMode ? 'text-yellow-400 fill-yellow-400' : ''}`} />
              <span className="hidden lg:inline uppercase tracking-widest">Neural</span>
            </button>
            <button onClick={toggleTheme} className="p-2.5 sm:p-3 rounded-2xl bg-white dark:bg-slate-800 text-slate-600 dark:text-slate-400 border border-slate-100 dark:border-slate-700">
              {theme === 'light' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
            </button>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-10 relative z-10 pb-40">
        {mode === AppMode.CAMERA && (
          <div className="space-y-12">
            <div className="max-w-2xl mx-auto text-center space-y-4">
              <h2 className="text-4xl md:text-6xl font-black text-slate-900 dark:text-white tracking-tighter">
                {responseMode === ResponseMode.SIMPLE ? "Resposta" : "Análise"}{" "}
                <span className="text-brand-600 dark:text-brand-400">Deep Resolve.</span>
              </h2>
            </div>
            
            <div className="max-w-3xl mx-auto">
              <CameraCapture onCapture={handleCapture} isProcessing={isSolving} />
            </div>

            {isSolving && (
              <div className="flex flex-col items-center justify-center p-20 space-y-8">
                <div className="relative w-20 h-20">
                  <div className="absolute inset-0 bg-brand-500/20 blur-2xl animate-pulse rounded-full"></div>
                  <Loader2 className="w-full h-full text-brand-600 dark:text-brand-400 animate-spin relative z-10" />
                </div>
                <p className="text-slate-500 dark:text-slate-400 font-bold italic animate-pulse text-sm">Escaneando fórmula...</p>
              </div>
            )}

            <div className="max-w-4xl mx-auto space-y-8 mt-12">
              {activeSolutions.map((solution, index) => (
                <React.Fragment key={solution.id}>
                  <SolutionDisplay content={solution.answer} isThinking={isThinkingMode} />
                  {(index + 1) % 2 === 0 && <AdPlaceholder />}
                </React.Fragment>
              ))}
            </div>
          </div>
        )}

        {mode === AppMode.CHAT && (
          <div className="max-w-3xl mx-auto space-y-12">
            <form onSubmit={handleChat} className="relative group">
              <div className="absolute -inset-1 bg-gradient-to-r from-brand-600 to-indigo-600 rounded-[2rem] blur opacity-20"></div>
              <input 
                type="text"
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                placeholder="Digite sua dúvida..."
                className="relative w-full p-6 pr-20 bg-white dark:bg-slate-900 rounded-[1.5rem] border-2 border-slate-100 dark:border-slate-800 focus:border-brand-500 outline-none shadow-xl transition-all font-semibold"
              />
              <button type="submit" className="absolute right-4 top-4 bottom-4 px-5 bg-brand-600 text-white rounded-xl shadow-lg">
                <ArrowRight className="w-6 h-6" />
              </button>
            </form>

            <div className="space-y-8">
              {activeSolutions.map((solution, index) => (
                <React.Fragment key={solution.id}>
                    <SolutionDisplay content={solution.answer} isThinking={isThinkingMode} />
                    {(index + 1) % 3 === 0 && <AdPlaceholder />}
                </React.Fragment>
              ))}
            </div>
          </div>
        )}

        {mode === AppMode.VOICE && <VoiceAssistant />}

        {mode === AppMode.HISTORY && (
          <div className="max-w-4xl mx-auto space-y-6">
            <h2 className="text-3xl font-black mb-8">Histórico Neural</h2>
            {history.map(item => (
              <div key={item.id} onClick={() => { setActiveSolutions([item]); setMode(AppMode.CAMERA); }} className="bg-white dark:bg-slate-900 p-6 rounded-3xl shadow-sm border border-slate-100 dark:border-slate-800 flex items-center justify-between cursor-pointer">
                <div className="flex items-center gap-4">
                  <div className="w-16 h-16 rounded-2xl bg-brand-50 dark:bg-brand-900/30 flex items-center justify-center text-brand-600">
                    {item.image ? <CameraIcon className="w-8 h-8"/> : <MessageSquare className="w-8 h-8"/>}
                  </div>
                  <div>
                    <p className="font-bold text-slate-900 dark:text-slate-100 line-clamp-1">{item.answer}</p>
                    <span className="text-[10px] text-slate-400 font-black uppercase tracking-widest">{new Date(item.timestamp).toLocaleDateString()}</span>
                  </div>
                </div>
                <ChevronRight className="w-6 h-6 text-slate-300" />
              </div>
            ))}
          </div>
        )}
      </main>

      <footer className="fixed bottom-0 left-0 right-0 z-[100] safe-bottom pb-6 px-6 pointer-events-none">
        <div className="max-w-md mx-auto pointer-events-auto">
          <nav className="glass dark:bg-slate-900/95 border border-white/20 dark:border-slate-800 rounded-[2.5rem] shadow-2xl p-4 flex items-center justify-between">
            <NavButton active={mode === AppMode.CAMERA} onClick={() => setMode(AppMode.CAMERA)} icon={<CameraIcon />} label="Scanner" />
            <NavButton active={mode === AppMode.CHAT} onClick={() => setMode(AppMode.CHAT)} icon={<MessageSquare />} label="Chat" />
            <NavButton active={mode === AppMode.VOICE} onClick={() => setMode(AppMode.VOICE)} icon={<Mic />} label="Voz" />
            <NavButton active={mode === AppMode.HISTORY} onClick={() => setMode(AppMode.HISTORY)} icon={<HistoryIcon />} label="Logs" />
          </nav>
        </div>
      </footer>
      {/* Safe Area Inset para o rodapé em Apps Nativo */}
      <div className="h-[env(safe-area-inset-bottom)] bg-transparent"></div>
    </div>
  );
};

const NavButton = ({ active, onClick, icon, label }: any) => (
  <button onClick={onClick} className={`flex flex-col items-center gap-1 transition-all ${active ? 'text-brand-600 dark:text-brand-400' : 'text-slate-400'}`}>
    <div className={`p-3 rounded-2xl transition-all ${active ? 'bg-brand-600 text-white shadow-lg shadow-brand-600/30 scale-110' : 'hover:bg-slate-100 dark:hover:bg-slate-800'}`}>
      {React.cloneElement(icon as React.ReactElement, { className: 'w-6 h-6' })}
    </div>
    <span className="text-[9px] font-black uppercase tracking-tighter">{label}</span>
  </button>
);

export default App;
