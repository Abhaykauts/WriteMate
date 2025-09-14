import React, { useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { motion } from 'framer-motion';
import { 
  Menu, 
  X, 
  Sun, 
  Moon, 
  User, 
  Settings, 
  LogOut,
  Mic,
  Hand,
  PenTool
} from 'lucide-react';
import { useAuth } from '../../context/AuthContext';

const Navbar = () => {
  const { 
    user, 
    signOut, 
    darkMode, 
    toggleDarkMode, 
    voiceEnabled, 
    setVoiceEnabled, 
    gestureEnabled, 
    setGestureEnabled 
  } = useAuth();
  const [isMenuOpen, setIsMenuOpen] = useState(false);
  const [isUserMenuOpen, setIsUserMenuOpen] = useState(false);
  const location = useLocation();

  const isActive = (path) => location.pathname === path;

  const handleLogout = () => {
    signOut();
    setIsUserMenuOpen(false);
  };

  const toggleVoice = () => setVoiceEnabled(!voiceEnabled);
  const toggleGesture = () => setGestureEnabled(!gestureEnabled);

  return (
    <motion.nav
      initial={{ y: -100, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6 }}
      className={`fixed top-0 left-0 right-0 z-50 backdrop-blur-md border-b transition-all duration-300 ${
        darkMode 
          ? 'bg-gray-900/80 border-gray-700/50' 
          : 'bg-white/80 border-white/20'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-3 group">
            <motion.div
              whileHover={{ scale: 1.1, rotate: 5 }}
              whileTap={{ scale: 0.95 }}
              className={`w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-300 ${
                darkMode
                  ? 'bg-gradient-to-br from-orange-500 to-pink-600'
                  : 'bg-gradient-to-br from-orange-400 to-pink-500'
              }`}
            >
              <PenTool className="w-5 h-5 text-white" />
            </motion.div>
            <motion.span
              className={`text-2xl font-bold bg-gradient-to-r from-orange-500 to-pink-600 bg-clip-text text-transparent ${ darkMode ? 'opacity-90' : '' }`}
              whileHover={{ scale: 1.05 }}
            >
              WriteMate
            </motion.span>
          </Link>

          {/* Desktop Navigation (when logged in) */}
          {user && (
            <>
              <NavLink to="/dashboard" isActive={isActive} darkMode={darkMode}>
                Dashboard
              </NavLink>
              <NavLink to="/write" isActive={isActive} darkMode={darkMode}>
                Write
              </NavLink>
              <NavLink to="/documents" isActive={isActive} darkMode={darkMode}>
                Documents
              </NavLink>
            </>
          )}

          <div className="hidden md:flex items-center space-x-8">
            <NavLink to="/about" isActive={isActive} darkMode={darkMode}>
              About Us
            </NavLink>
            <NavLink to="/contact" isActive={isActive} darkMode={darkMode}>
              Contact Us
            </NavLink>
          </div>

          {/* Controls and User Menu */}
          <div className="flex items-center space-x-4">
            {/* Voice & Gesture (only logged in) */}
            {user && (
              <div className="hidden md:flex items-center space-x-2">
                <IconButton
                  onClick={toggleVoice}
                  active={voiceEnabled}
                  darkMode={darkMode}
                  icon={<Mic className={`w-4 h-4 ${voiceEnabled ? 'animate-pulse' : ''}`} />}
                  title="Voice Control"
                />
                <IconButton
                  onClick={toggleGesture}
                  active={gestureEnabled}
                  darkMode={darkMode}
                  icon={<Hand className={`w-4 h-4 ${gestureEnabled ? 'animate-pulse' : ''}`} />}
                  title="Gesture Control"
                />
              </div>
            )}

            {/* Dark Mode Toggle */}
            <IconButton
              onClick={toggleDarkMode}
              darkMode={darkMode}
              icon={darkMode ? <Sun className="w-4 h-4 text-yellow-400" /> : <Moon className="w-4 h-4" />}
              title={darkMode ? 'Light Mode' : 'Dark Mode'}
            />

            {/* User Menu / Sign In */}
            {user ? (
              <UserMenu
                user={user}
                darkMode={darkMode}
                isUserMenuOpen={isUserMenuOpen}
                setIsUserMenuOpen={setIsUserMenuOpen}
                handleLogout={handleLogout}
              />
            ) : (
              <Link
                to="/signin"
                className="bg-gradient-to-r from-orange-500 to-pink-600 text-white px-6 py-2 rounded-lg font-medium hover:shadow-lg transition-all duration-300 transform hover:scale-105"
              >
                Sign In
              </Link>
            )}

            {/* Mobile Menu Toggle */}
            <motion.button
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className={`md:hidden p-2 rounded-lg transition-all duration-300 ${
                darkMode
                  ? 'bg-gray-800/50 text-white hover:bg-gray-700/50'
                  : 'bg-white/50 text-gray-600 hover:bg-white/80'
              }`}
            >
              {isMenuOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </motion.button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className={`md:hidden border-t mt-4 pt-4 pb-4 ${
              darkMode ? 'border-gray-700/50' : 'border-white/20'
            }`}
          >
            <div className="flex flex-col space-y-2">
              <NavLink to="/about" isActive={isActive} darkMode={darkMode} mobile>
                About Us
              </NavLink>
              <NavLink to="/contact" isActive={isActive} darkMode={darkMode} mobile>
                Contact Us
              </NavLink>
              {user && (
                <>
                  <NavLink to="/dashboard" isActive={isActive} darkMode={darkMode} mobile>
                    Dashboard
                  </NavLink>
                  <NavLink to="/write" isActive={isActive} darkMode={darkMode} mobile>
                    Write
                  </NavLink>
                  <NavLink to="/documents" isActive={isActive} darkMode={darkMode} mobile>
                    Documents
                  </NavLink>
                  {/* Mobile Voice/Gesture */}
                  <div className="flex items-center space-x-2 px-4 py-2">
                    <IconButton
                      onClick={toggleVoice}
                      active={voiceEnabled}
                      darkMode={darkMode}
                      icon={<Mic className={`w-4 h-4 mx-auto ${voiceEnabled ? 'animate-pulse' : ''}`} />}
                    />
                    <IconButton
                      onClick={toggleGesture}
                      active={gestureEnabled}
                      darkMode={darkMode}
                      icon={<Hand className={`w-4 h-4 mx-auto ${gestureEnabled ? 'animate-pulse' : ''}`} />}
                    />
                  </div>
                </>
              )}
            </div>
          </motion.div>
        )}
      </div>
    </motion.nav>
  );
};

/* ---------- Sub Components ---------- */

const NavLink = ({ to, isActive, darkMode, children, mobile }) => (
  <Link
    to={to}
    className={`px-4 py-2 rounded-lg transition-all duration-300 ${
      isActive(to)
        ? darkMode
          ? 'bg-orange-500/20 text-orange-400'
          : 'bg-orange-100 text-orange-600'
        : darkMode
        ? 'text-white hover:text-orange-400 hover:bg-gray-800/50'
        : 'text-gray-700 hover:text-orange-600 hover:bg-orange-50'
    } ${mobile ? 'block' : ''}`}
  >
    {children}
  </Link>
);

const IconButton = ({ onClick, icon, active, darkMode, title }) => (
  <motion.button
    whileHover={{ scale: 1.05 }}
    whileTap={{ scale: 0.95 }}
    onClick={onClick}
    title={title}
    className={`p-2 rounded-lg transition-all duration-300 ${
      active
        ? 'bg-gradient-to-r from-orange-500 to-pink-600 text-white shadow-lg'
        : darkMode
        ? 'bg-gray-800/50 text-white hover:bg-gray-700/50'
        : 'bg-white/50 text-gray-600 hover:bg-white/80'
    }`}
  >
    {icon}
  </motion.button>
);

const UserMenu = ({ user, darkMode, isUserMenuOpen, setIsUserMenuOpen, handleLogout }) => (
  <div className="relative">
    <motion.button
      whileHover={{ scale: 1.05 }}
      whileTap={{ scale: 0.95 }}
      onClick={() => setIsUserMenuOpen(!isUserMenuOpen)}
      className={`flex items-center space-x-2 p-2 rounded-lg transition-all duration-300 ${
        darkMode
          ? 'bg-gray-800/50 text-white hover:bg-gray-700/50'
          : 'bg-white/50 text-gray-600 hover:bg-white/80'
      }`}
    >
      <div className="w-8 h-8 bg-gradient-to-br from-orange-400 to-pink-500 rounded-full flex items-center justify-center">
        <User className="w-4 h-4 text-white" />
      </div>
      <span className="hidden sm:block font-medium">
        {user.email?.split('@')[0] || 'User'}
      </span>
    </motion.button>

    {/* Dropdown */}
    {isUserMenuOpen && (
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={`absolute right-0 mt-2 w-48 rounded-xl shadow-lg backdrop-blur-md border transition-all duration-300 ${
          darkMode
            ? 'bg-gray-900/90 border-gray-700/50'
            : 'bg-white/90 border-white/20'
        }`}
      >
        <div className="py-2">
          <Link
            to="/profile"
            onClick={() => setIsUserMenuOpen(false)}
            className={`flex items-center space-x-2 px-4 py-2 transition-colors ${
              darkMode
                ? 'text-white hover:bg-gray-800/50'
                : 'text-gray-700 hover:bg-gray-50'
            }`}
          >
            <User className="w-4 h-4" />
            <span>Profile</span>
          </Link>
          <Link
            to="/settings"
            onClick={() => setIsUserMenuOpen(false)}
            className={`flex items-center space-x-2 px-4 py-2 transition-colors ${
              darkMode
                ? 'text-white hover:bg-gray-800/50'
                : 'text-gray-700 hover:bg-gray-50'
            }`}
          >
            <Settings className="w-4 h-4" />
            <span>Settings</span>
          </Link>
          <button
            onClick={handleLogout}
            className={`flex items-center space-x-2 w-full px-4 py-2 text-left transition-colors ${
              darkMode
                ? 'text-red-400 hover:bg-red-900/20'
                : 'text-red-600 hover:bg-red-50'
            }`}
          >
            <LogOut className="w-4 h-4" />
            <span>Sign Out</span>
          </button>
        </div>
      </motion.div>
    )}
  </div>
);

export default Navbar;
