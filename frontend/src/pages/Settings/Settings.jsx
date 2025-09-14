import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  MicrophoneIcon,
  HandRaisedIcon,
  SparklesIcon,
  BellIcon,
  Cog6ToothIcon,
  SpeakerWaveIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../../context/AuthContext';

const Settings = () => {
  const { darkMode } = useAuth();

  const [voiceSettings, setVoiceSettings] = useState({
    enabled: true,
    sensitivity: 75
  });

  const [gestureSettings, setGestureSettings] = useState({
    enabled: true,
    sensitivity: 50
  });

  const [aiSettings, setAiSettings] = useState({
    enabled: true,
    frequency: 60
  });

  const [notificationSettings, setNotificationSettings] = useState({
    enabled: true,
    email: true,
    sound: true,
    volume: 80
  });

  const [generalSettings, setGeneralSettings] = useState({
    autoSave: true,
    theme: 'bright'
  });

  return (
    <div className={`min-h-screen pt-24 pb-8 px-4 sm:px-6 lg:px-8 transition-colors duration-300 ${
      darkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          className="flex items-center justify-between mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <h1 className="text-4xl font-bold mb-2">
              <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">
                Settings
              </span>
            </h1>
            <p className={`text-xl ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              Customize your WriteMate experience
            </p>
          </div>
          <button className="px-6 py-3 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-full font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 shadow-lg">
            Save Changes
          </button>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Voice Input Settings */}
          <SettingsCard icon={MicrophoneIcon} title="Voice Input" delay={0.1} darkMode={darkMode}>
            <div className="space-y-6">
              <ToggleSwitch
                label="Enable Voice Input"
                checked={voiceSettings.enabled}
                onChange={(checked) => setVoiceSettings({...voiceSettings, enabled: checked})}
                darkMode={darkMode}
              />
              
              <SliderControl
                label="Voice Sensitivity"
                value={voiceSettings.sensitivity}
                onChange={(value) => setVoiceSettings({...voiceSettings, sensitivity: value})}
                min={0}
                max={100}
                unit="%"
                marks={['Low', '75%', 'High']}
                darkMode={darkMode}
              />

              <div className={`rounded-lg p-4 ${
                darkMode ? 'bg-blue-900/30' : 'bg-blue-50'
              }`}>
                <h4 className={`font-semibold mb-2 ${
                  darkMode ? 'text-blue-300' : 'text-blue-800'
                }`}>
                  Voice Commands
                </h4>
                <ul className={`text-sm space-y-1 ${
                  darkMode ? 'text-blue-200' : 'text-blue-700'
                }`}>
                  <li>👆 <span className="font-medium">"Start writing"</span> - Begin voice input</li>
                  <li>👆 <span className="font-medium">"New paragraph"</span> - Add paragraph break</li>
                  <li>👆 <span className="font-medium">"Delete that"</span> - Remove last phrase</li>
                </ul>
              </div>
            </div>
          </SettingsCard>

          {/* Gesture Control Settings */}
          <SettingsCard icon={HandRaisedIcon} title="Gesture Control" delay={0.2} darkMode={darkMode}>
            <div className="space-y-6">
              <ToggleSwitch
                label="Enable Gesture Control"
                checked={gestureSettings.enabled}
                onChange={(checked) => setGestureSettings({...gestureSettings, enabled: checked})}
                darkMode={darkMode}
              />
              
              <SliderControl
                label="Gesture Sensitivity"
                value={gestureSettings.sensitivity}
                onChange={(value) => setGestureSettings({...gestureSettings, sensitivity: value})}
                min={0}
                max={100}
                unit="%"
                marks={['Precise', '50%', 'Loose']}
                darkMode={darkMode}
              />

              <div className={`rounded-lg p-4 ${
                darkMode ? 'bg-purple-900/30' : 'bg-purple-50'
              }`}>
                <h4 className={`font-semibold mb-2 ${
                  darkMode ? 'text-purple-300' : 'text-purple-800'
                }`}>
                  Gesture Guide
                </h4>
                <ul className={`text-sm space-y-1 ${
                  darkMode ? 'text-purple-200' : 'text-purple-700'
                }`}>
                  <li>👆 Point up - Scroll up</li>
                  <li>👇 Point down - Scroll down</li>
                  <li>✋ Open palm - Pause/Stop</li>
                </ul>
              </div>
            </div>
          </SettingsCard>

          {/* AI Writing Assistant */}
          <SettingsCard icon={SparklesIcon} title="AI Writing Assistant" delay={0.3} darkMode={darkMode}>
            <div className="space-y-6">
              <ToggleSwitch
                label="Enable AI Suggestions"
                checked={aiSettings.enabled}
                onChange={(checked) => setAiSettings({...aiSettings, enabled: checked})}
                darkMode={darkMode}
              />
              
              <SliderControl
                label="Suggestion Frequency"
                value={aiSettings.frequency}
                onChange={(value) => setAiSettings({...aiSettings, frequency: value})}
                min={0}
                max={100}
                unit="%"
                marks={['Minimal', '60%', 'Frequent']}
                darkMode={darkMode}
              />
            </div>
          </SettingsCard>

          {/* Notifications */}
          <SettingsCard icon={BellIcon} title="Notifications" delay={0.4} darkMode={darkMode}>
            <div className="space-y-6">
              <ToggleSwitch
                label="Enable Notifications"
                checked={notificationSettings.enabled}
                onChange={(checked) => setNotificationSettings({...notificationSettings, enabled: checked})}
                darkMode={darkMode}
              />
              
              <ToggleSwitch
                label="Email Notifications"
                checked={notificationSettings.email}
                onChange={(checked) => setNotificationSettings({...notificationSettings, email: checked})}
                darkMode={darkMode}
              />

              <ToggleSwitch
                label="Sound Effects"
                checked={notificationSettings.sound}
                onChange={(checked) => setNotificationSettings({...notificationSettings, sound: checked})}
                darkMode={darkMode}
              />

              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className={`font-medium flex items-center ${
                    darkMode ? 'text-gray-200' : 'text-gray-700'
                  }`}>
                    <SpeakerWaveIcon className="w-4 h-4 mr-2" />
                    Volume
                  </label>
                </div>
                <div className="relative">
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={notificationSettings.volume}
                    onChange={(e) => setNotificationSettings({...notificationSettings, volume: parseInt(e.target.value)})}
                    className="w-full h-2 bg-gray-200 rounded-full appearance-none cursor-pointer slider"
                  />
                  <div className={`flex justify-between text-xs mt-1 ${
                    darkMode ? 'text-gray-400' : 'text-gray-500'
                  }`}>
                    <span>0%</span>
                    <span className="font-medium">{notificationSettings.volume}%</span>
                    <span>100%</span>
                  </div>
                </div>
              </div>
            </div>
          </SettingsCard>
        </div>

        {/* General Settings */}
        <motion.div
          className="mt-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <SettingsCard icon={Cog6ToothIcon} title="General Settings" fullWidth darkMode={darkMode}>
            <div className="grid md:grid-cols-2 gap-8">
              <div className="space-y-6">
                <ToggleSwitch
                  label="Auto-save Documents"
                  checked={generalSettings.autoSave}
                  onChange={(checked) => setGeneralSettings({...generalSettings, autoSave: checked})}
                  darkMode={darkMode}
                />
              </div>

              <div className="space-y-6">
                <div>
                  <label className={`block font-medium mb-3 ${
                    darkMode ? 'text-gray-200' : 'text-gray-700'
                  }`}>
                    Theme
                  </label>
                  <div className={`flex items-center justify-between p-3 rounded-lg border ${
                    darkMode ? 'bg-gray-800 border-gray-700' : 'bg-white/50 border-white/20'
                  }`}>
                    <span className={darkMode ? 'text-gray-300' : 'text-gray-600'}>
                      Bright theme active
                    </span>
                    <button className={`px-4 py-2 rounded-lg transition-colors ${
                      darkMode ? 'bg-gray-700 text-gray-200 hover:bg-gray-600' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}>
                      Customize
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </SettingsCard>
        </motion.div>

        {/* Save All Button */}
        <motion.div
          className="mt-8 text-center"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.6 }}
        >
          <button className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-full text-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 shadow-lg">
            Save All Settings
          </button>
        </motion.div>
      </div>
    </div>
  );
};

/* ---------- Sub Components ---------- */

const SettingsCard = ({ icon: Icon, title, children, delay = 0, fullWidth = false, darkMode }) => (
  <motion.div
    className={`backdrop-blur-md rounded-2xl p-6 border transition-colors duration-300 ${
      darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white/30 border-white/20'
    } ${fullWidth ? 'lg:col-span-2' : ''}`}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
  >
    <h2 className={`text-xl font-bold mb-6 flex items-center ${
      darkMode ? 'text-white' : 'text-gray-800'
    }`}>
      <Icon className="w-6 h-6 mr-2" />
      {title}
    </h2>
    {children}
  </motion.div>
);

const ToggleSwitch = ({ label, checked, onChange, darkMode }) => (
  <div className="flex items-center justify-between">
    <label className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
      {label}
    </label>
    <button
      onClick={() => onChange(!checked)}
      className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
        checked ? 'bg-orange-500' : 'bg-gray-300'
      }`}
    >
      <span
        className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
          checked ? 'translate-x-6' : 'translate-x-1'
        }`}
      />
    </button>
  </div>
);

const SliderControl = ({ label, value, onChange, min, max, unit, marks, darkMode }) => (
  <div className="space-y-3">
    <div className="flex items-center justify-between">
      <label className={`font-medium ${darkMode ? 'text-gray-200' : 'text-gray-700'}`}>
        {label}
      </label>
      <span className={darkMode ? 'text-gray-300 font-semibold' : 'text-gray-600 font-semibold'}>
        {value}{unit}
      </span>
    </div>
    <div className="relative">
      <input
        type="range"
        min={min}
        max={max}
        value={value}
        onChange={(e) => onChange(parseInt(e.target.value))}
        className="w-full h-2 bg-gray-200 rounded-full appearance-none cursor-pointer slider"
      />
      {marks && (
        <div className={`flex justify-between text-xs mt-1 ${
          darkMode ? 'text-gray-400' : 'text-gray-500'
        }`}>
          {marks.map((mark, index) => (
            <span key={index}>{mark}</span>
          ))}
        </div>
      )}
    </div>
  </div>
);

export default Settings;
