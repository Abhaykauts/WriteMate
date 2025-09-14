import React, { useState } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';

const SignIn = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const { signIn, darkMode } = useAuth(); // ✅ get darkMode
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      await signIn(email, password);
      navigate('/dashboard');
    } catch (error) {
      console.error('Sign in error:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div
      className={`min-h-screen flex items-center justify-center py-12 px-4 sm:px-6 lg:px-8 transition-colors duration-300 ${
        darkMode ? 'bg-gray-900 text-white' : 'bg-gray-50 text-gray-900'
      }`}
    >
      <div className="max-w-md w-full">
        {/* Logo */}
        <motion.div
          className="text-center mb-8"
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <Link to="/" className="flex items-center justify-center space-x-2">
            <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-pink-500 rounded-xl flex items-center justify-center">
              <span className="text-white font-bold text-xl">W</span>
            </div>
            <span className="text-3xl font-bold bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">
              WriteMate
            </span>
          </Link>
        </motion.div>

        {/* Sign In Form */}
        <motion.div
          className={`backdrop-blur-md rounded-2xl p-8 border transition-colors duration-300 ${
            darkMode
              ? 'bg-gray-800/50 border-gray-700'
              : 'bg-white/30 border-white/20'
          }`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="text-center mb-8">
            <h2
              className={`text-3xl font-bold mb-2 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}
            >
              Welcome back
            </h2>
            <p className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
              Sign in to continue to WriteMate
            </p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label
                htmlFor="email"
                className={`block text-sm font-medium mb-2 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}
              >
                Email address
              </label>
              <input
                id="email"
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className={`w-full px-4 py-3 rounded-lg outline-none border transition-all duration-200 ${
                  darkMode
                    ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:bg-gray-600 focus:border-orange-400'
                    : 'bg-white/50 border-white/20 text-gray-900 placeholder-gray-500 focus:bg-white/70 focus:border-orange-300'
                }`}
                placeholder="Enter your email"
              />
            </div>

            <div>
              <label
                htmlFor="password"
                className={`block text-sm font-medium mb-2 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}
              >
                Password
              </label>
              <div className="relative">
                <input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  required
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className={`w-full px-4 py-3 rounded-lg outline-none border transition-all duration-200 pr-12 ${
                    darkMode
                      ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400 focus:bg-gray-600 focus:border-orange-400'
                      : 'bg-white/50 border-white/20 text-gray-900 placeholder-gray-500 focus:bg-white/70 focus:border-orange-300'
                  }`}
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className={`absolute right-3 top-1/2 transform -translate-y-1/2 ${
                    darkMode
                      ? 'text-gray-400 hover:text-gray-200'
                      : 'text-gray-500 hover:text-gray-700'
                  }`}
                >
                  {showPassword ? (
                    <EyeSlashIcon className="w-5 h-5" />
                  ) : (
                    <EyeIcon className="w-5 h-5" />
                  )}
                </button>
              </div>
            </div>

            <div className="flex items-center justify-between">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  className={`w-4 h-4 text-orange-500 rounded focus:ring-orange-500 ${
                    darkMode ? 'bg-gray-700 border-gray-600' : 'border-gray-300'
                  }`}
                />
                <span
                  className={`ml-2 text-sm ${
                    darkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}
                >
                  Remember me
                </span>
              </label>
              <Link
                to="/forgot-password"
                className="text-sm text-orange-500 hover:text-orange-600"
              >
                Forgot password?
              </Link>
            </div>

            <button
              type="submit"
              disabled={loading}
              className="w-full flex items-center justify-center px-8 py-3 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? (
                <div className="w-6 h-6 border-2 border-white border-t-transparent rounded-full animate-spin" />
              ) : (
                'Sign in'
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <p className={darkMode ? 'text-gray-400' : 'text-gray-600'}>
              Don&apos;t have an account?{' '}
              <Link
                to="/signup"
                className="text-orange-500 hover:text-orange-600 font-semibold"
              >
                Sign up
              </Link>
            </p>
          </div>
        </motion.div>

        {/* Features */}
        <motion.div
          className="mt-8 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
        >
          <p className={darkMode ? 'text-gray-400 mb-4' : 'text-gray-600 mb-4'}>
            Experience the future of writing with:
          </p>
          <div
            className={`flex justify-center space-x-8 text-sm ${
              darkMode ? 'text-gray-300' : 'text-gray-700'
            }`}
          >
            <div className="flex items-center space-x-2">
              <span className="text-red-500">🎤</span>
              <span>Voice Control</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-blue-500">✋</span>
              <span>Gesture Control</span>
            </div>
            <div className="flex items-center space-x-2">
              <span className="text-purple-500">✨</span>
              <span>AI Assistant</span>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default SignIn;
