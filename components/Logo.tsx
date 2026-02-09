
import React from 'react';

const Logo: React.FC<{ className?: string }> = ({ className = "w-10 h-10" }) => {
  return (
    <div className={`flex items-center justify-center rounded-xl bg-white dark:bg-slate-800 shadow-sm p-2 transition-colors ${className}`}>
      <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" className="w-full h-full text-brand-600 dark:text-brand-400">
        <path d="M12 2L2 7L12 12L22 7L12 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M2 17L12 22L22 17" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M2 12L12 17L22 12" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <path d="M12 12V22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        <circle cx="12" cy="7" r="1.5" fill="currentColor" />
      </svg>
    </div>
  );
};

export default Logo;
