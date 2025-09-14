import React from 'react';
import { Link } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';

const Footer = () => {
  const { darkMode } = useAuth();

  return (
    <footer
      className={`backdrop-blur-md border-t py-6 transition-colors duration-300 ${
        darkMode
          ? 'bg-gray-900/50 border-gray-700'
          : 'bg-white/20 border-white/20'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Logo */}
        <div className="flex items-center justify-center">
          <div className="flex items-center space-x-2">
            <div className="w-6 h-6 bg-gradient-to-r from-orange-500 to-pink-500 rounded flex items-center justify-center">
              <span className="text-white font-bold text-xs">W</span>
            </div>
            <span className="text-lg font-bold bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">
              WriteMate
            </span>
          </div>
        </div>

        {/* Text */}
        <div className="text-center mt-4">
          <p
            className={`text-sm ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}
          >
            © 2025 WriteMate. Making writing smarter and more accessible.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
