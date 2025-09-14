import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import {
  MicrophoneIcon,
  HandRaisedIcon,
  SpeakerWaveIcon,
  SparklesIcon,
  DocumentTextIcon,
  ClockIcon,
  BookOpenIcon
} from '@heroicons/react/24/outline';

const Write = () => {
  const { voiceEnabled, gestureEnabled, setVoiceEnabled, setGestureEnabled, darkMode } = useAuth();
  const [documentTitle, setDocumentTitle] = useState('');
  const [content, setContent] = useState('');

  const wordCount = content.split(/\s+/).filter(word => word.length > 0).length;
  const characterCount = content.length;
  const readingTime = Math.ceil(wordCount / 200);

  const toggleVoice = () => {
    setVoiceEnabled(!voiceEnabled);
  };

  const toggleGesture = () => {
    setGestureEnabled(!gestureEnabled);
  };

  return (
    <div
      className={`min-h-screen pt-24 pb-8 px-4 sm:px-6 lg:px-8 transition-colors duration-300 ${
        darkMode ? 'bg-gray-900' : 'bg-gray-50'
      }`}
    >
      <div className="max-w-7xl mx-auto">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Main Writing Area */}
          <div className="flex-1">
            <motion.div
              className={`backdrop-blur-md rounded-2xl border transition-colors duration-300 ${
                darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white/30 border-white/20'
              }`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
            >
              {/* Header */}
              <div
                className={`p-6 border-b transition-colors ${
                  darkMode ? 'border-gray-700' : 'border-white/20'
                }`}
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h1
                      className={`text-3xl font-bold mb-2 ${
                        darkMode ? 'text-white' : 'text-gray-800'
                      }`}
                    >
                      <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">
                        Write
                      </span>{' '}
                      Mode
                    </h1>
                    <p className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                      Express your ideas with voice, gestures, or keyboard
                    </p>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                      {wordCount} words
                    </span>
                    <button className="px-6 py-2 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-full font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200">
                      Save Document
                    </button>
                  </div>
                </div>
              </div>

              {/* Document Title */}
              <div
                className={`p-6 border-b transition-colors ${
                  darkMode ? 'border-gray-700' : 'border-white/20'
                }`}
              >
                <input
                  type="text"
                  placeholder="Enter document title..."
                  value={documentTitle}
                  onChange={(e) => setDocumentTitle(e.target.value)}
                  className={`w-full text-2xl font-semibold bg-transparent border-none outline-none placeholder-gray-400 ${
                    darkMode ? 'text-white' : 'text-gray-800'
                  }`}
                />
              </div>

              {/* Writing Area */}
              <div className="p-6">
                <textarea
                  placeholder="Start writing... Use voice commands or gestures for a hands-free experience."
                  value={content}
                  onChange={(e) => setContent(e.target.value)}
                  className={`w-full h-96 text-lg bg-transparent border-none outline-none resize-none leading-relaxed placeholder-gray-400 ${
                    darkMode ? 'text-white' : 'text-gray-800'
                  }`}
                />
              </div>
            </motion.div>
          </div>

          {/* Sidebar Controls */}
          <div className="lg:w-80 space-y-6">
            {/* Input Controls */}
            <motion.div
              className={`backdrop-blur-md rounded-2xl p-6 border transition-colors duration-300 ${
                darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white/30 border-white/20'
              }`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
            >
              <h3
                className={`text-xl font-bold mb-4 flex items-center ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}
              >
                <SparklesIcon className="w-5 h-5 mr-2" />
                Input Controls
              </h3>

              <div className="space-y-4">
                <ControlButton
                  active={voiceEnabled}
                  onClick={toggleVoice}
                  icon={MicrophoneIcon}
                  label="Start Voice"
                  enabled={true}
                  darkMode={darkMode}
                />
                <ControlButton
                  active={gestureEnabled}
                  onClick={toggleGesture}
                  icon={HandRaisedIcon}
                  label="Start Gestures"
                  enabled={true}
                  darkMode={darkMode}
                />
              </div>

              <div
                className={`mt-6 pt-6 border-t ${
                  darkMode ? 'border-gray-700' : 'border-white/20'
                }`}
              >
                <h4
                  className={`font-semibold mb-3 ${
                    darkMode ? 'text-white' : 'text-gray-800'
                  }`}
                >
                  Quick Actions:
                </h4>
                <div className="space-y-2">
                  <ActionButton icon={SpeakerWaveIcon} label="Read Aloud" darkMode={darkMode} />
                  <ActionButton icon={SparklesIcon} label="AI Suggest" darkMode={darkMode} />
                </div>
              </div>
            </motion.div>

            {/* Writing Stats */}
            <motion.div
              className={`backdrop-blur-md rounded-2xl p-6 border transition-colors duration-300 ${
                darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white/30 border-white/20'
              }`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.3 }}
            >
              <h3
                className={`text-xl font-bold mb-4 flex items-center ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}
              >
                <DocumentTextIcon className="w-5 h-5 mr-2" />
                Writing Stats
              </h3>

              <div className="space-y-4">
                <StatItem label="Words:" value={wordCount} darkMode={darkMode} />
                <StatItem label="Characters:" value={characterCount} darkMode={darkMode} />
                <StatItem label="Reading time:" value={`${readingTime} min`} darkMode={darkMode} />

                <div
                  className={`pt-4 border-t ${
                    darkMode ? 'border-gray-700' : 'border-white/20'
                  }`}
                >
                  <div className="flex justify-between items-center mb-2">
                    <span className={darkMode ? 'text-gray-300 font-medium' : 'text-gray-700 font-medium'}>
                      Daily Goal
                    </span>
                    <span className={darkMode ? 'text-white font-bold' : 'text-gray-700 font-bold'}>
                      0%
                    </span>
                  </div>
                  <div className={`text-sm mb-2 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
                    Target: 500 words
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className="bg-gradient-to-r from-orange-500 to-pink-500 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${Math.min((wordCount / 500) * 100, 100)}%` }}
                    />
                  </div>
                </div>
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ---------- Sub Components ---------- */

const ControlButton = ({ active, onClick, icon: Icon, label, enabled, darkMode }) => (
  <button
    onClick={onClick}
    disabled={!enabled}
    className={`w-full flex items-center justify-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 ${
      active && enabled
        ? 'bg-orange-500 text-white shadow-lg'
        : enabled
        ? darkMode
          ? 'bg-gray-700 text-gray-200 hover:bg-gray-600'
          : 'bg-white/50 text-gray-700 hover:bg-white/70'
        : 'bg-gray-200 text-gray-400 cursor-not-allowed'
    }`}
  >
    <Icon className="w-5 h-5" />
    <span className="font-medium">{label}</span>
  </button>
);

const ActionButton = ({ icon: Icon, label, darkMode }) => (
  <button
    className={`w-full flex items-center space-x-3 px-4 py-2 rounded-lg transition-colors ${
      darkMode
        ? 'bg-gray-700 text-gray-200 hover:bg-gray-600'
        : 'bg-white/50 text-gray-700 hover:bg-white/70'
    }`}
  >
    <Icon className="w-4 h-4" />
    <span className="font-medium">{label}</span>
  </button>
);

const StatItem = ({ label, value, darkMode }) => (
  <div className="flex justify-between items-center">
    <span className={darkMode ? 'text-gray-300 font-medium' : 'text-gray-700 font-medium'}>
      {label}
    </span>
    <span className={darkMode ? 'text-white font-bold' : 'text-gray-800 font-bold'}>{value}</span>
  </div>
);

export default Write;
