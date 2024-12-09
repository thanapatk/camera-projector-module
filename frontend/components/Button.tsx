import { ReactNode } from 'react';

interface ButtonProps {
  variant?: 'primary' | 'secondary';
  children: ReactNode;
  onClick?: () => void;
  className?: string;
}

export function Button({ 
  variant = 'primary', 
  children, 
  onClick,
  className = ''
}: ButtonProps) {
  const baseStyles = 'px-12 py-4 rounded-lg font-medium transition-colors min-w-[320px] drop-shadow-[0_2px_8px_rgba(255,255,255,0.10)]';
  const variants = {
    primary: 'bg-[#1133F0] text-white hover:bg-blue-700',
    secondary: 'bg-gray-600 text-white hover:bg-gray-700'
  };

  return (
    <button 
      className={`${baseStyles} ${variants[variant]} ${className}`}
      onClick={onClick}
    >
      {children}
    </button>
  );
}