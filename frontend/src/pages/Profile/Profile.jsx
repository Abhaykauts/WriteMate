import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import {
  UserIcon,
  KeyIcon,
  PhotoIcon,
  PencilIcon,
  CheckIcon,
  XMarkIcon
} from '@heroicons/react/24/outline';

const Profile = () => {
  const { user, darkMode } = useAuth();
  const [isEditing, setIsEditing] = useState(false);
  const [profileData, setProfileData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    bio: 'Passionate writer using AI technology to enhance creativity.',
    location: 'San Francisco, CA',
    website: 'https://writemate.com'
  });

  const handleSave = () => {
    setIsEditing(false);
    // Save logic here
  };

  const handleCancel = () => {
    setIsEditing(false);
    setProfileData({
      name: user?.name || '',
      email: user?.email || '',
      bio: 'Passionate writer using AI technology to enhance creativity.',
      location: 'San Francisco, CA',
      website: 'https://writemate.com'
    });
  };

  return (
    <div className="min-h-screen pt-24 pb-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <motion.div
          className="mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1
            className={`text-4xl font-bold mb-2 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}
          >
            <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">
              Profile
            </span>
          </h1>
          <p
            className={`text-xl ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}
          >
            Manage your account settings and preferences
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-3 gap-8">
          {/* Profile Card */}
          <motion.div
            className={`backdrop-blur-md rounded-2xl p-6 border lg:col-span-1 ${
              darkMode
                ? 'bg-gray-800/30 border-gray-700/20'
                : 'bg-white/30 border-white/20'
            }`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <div className="text-center">
              <div className="relative inline-block mb-4">
                <div className="w-24 h-24 bg-gradient-to-r from-orange-500 to-pink-500 rounded-full flex items-center justify-center text-white text-3xl font-bold">
                  {user?.avatar || 'U'}
                </div>
                <button className="absolute bottom-0 right-0 w-8 h-8 bg-white dark:bg-gray-700 rounded-full shadow-lg flex items-center justify-center hover:bg-gray-100 dark:hover:bg-gray-600 transition-colors">
                  <PhotoIcon className="w-4 h-4 text-gray-600 dark:text-gray-200" />
                </button>
              </div>

              <h2
                className={`text-2xl font-bold mb-1 ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}
              >
                {user?.name}
              </h2>
              <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} mb-4`}>
                {user?.email}
              </p>

              <div
                className={`space-y-3 text-sm ${
                  darkMode ? 'text-gray-400' : 'text-gray-600'
                }`}
              >
                <div className="flex items-center justify-center">
                  <span>🌍 {profileData.location}</span>
                </div>
                <div className="flex items-center justify-center">
                  <span>🔗 {profileData.website}</span>
                </div>
                <div className="flex items-center justify-center">
                  <span>📅 Joined December 2024</span>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Profile Information */}
          <motion.div
            className="lg:col-span-2 space-y-6"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {/* Personal Information */}
            <div
              className={`backdrop-blur-md rounded-2xl p-6 border ${
                darkMode
                  ? 'bg-gray-800/30 border-gray-700/20'
                  : 'bg-white/30 border-white/20'
              }`}
            >
              <div className="flex items-center justify-between mb-6">
                <h3
                  className={`text-xl font-bold flex items-center ${
                    darkMode ? 'text-white' : 'text-gray-800'
                  }`}
                >
                  <UserIcon className="w-5 h-5 mr-2" />
                  Personal Information
                </h3>
                {!isEditing ? (
                  <button
                    onClick={() => setIsEditing(true)}
                    className="flex items-center space-x-2 px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors"
                  >
                    <PencilIcon className="w-4 h-4" />
                    <span>Edit</span>
                  </button>
                ) : (
                  <div className="flex items-center space-x-2">
                    <button
                      onClick={handleSave}
                      className="flex items-center space-x-2 px-4 py-2 bg-green-500 text-white rounded-lg hover:bg-green-600 transition-colors"
                    >
                      <CheckIcon className="w-4 h-4" />
                      <span>Save</span>
                    </button>
                    <button
                      onClick={handleCancel}
                      className="flex items-center space-x-2 px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 transition-colors"
                    >
                      <XMarkIcon className="w-4 h-4" />
                      <span>Cancel</span>
                    </button>
                  </div>
                )}
              </div>

              <div className="grid md:grid-cols-2 gap-6">
                <FormField
                  label="Full Name"
                  value={profileData.name}
                  onChange={(value) => setProfileData({ ...profileData, name: value })}
                  disabled={!isEditing}
                  darkMode={darkMode}
                />
                <FormField
                  label="Email"
                  value={profileData.email}
                  onChange={(value) => setProfileData({ ...profileData, email: value })}
                  disabled={!isEditing}
                  darkMode={darkMode}
                />
                <FormField
                  label="Location"
                  value={profileData.location}
                  onChange={(value) => setProfileData({ ...profileData, location: value })}
                  disabled={!isEditing}
                  darkMode={darkMode}
                />
                <FormField
                  label="Website"
                  value={profileData.website}
                  onChange={(value) => setProfileData({ ...profileData, website: value })}
                  disabled={!isEditing}
                  darkMode={darkMode}
                />
              </div>

              <div className="mt-6">
                <FormField
                  label="Bio"
                  value={profileData.bio}
                  onChange={(value) => setProfileData({ ...profileData, bio: value })}
                  disabled={!isEditing}
                  multiline
                  darkMode={darkMode}
                />
              </div>
            </div>

            {/* Writing Statistics */}
            <div
              className={`backdrop-blur-md rounded-2xl p-6 border ${
                darkMode
                  ? 'bg-gray-800/30 border-gray-700/20'
                  : 'bg-white/30 border-white/20'
              }`}
            >
              <h3
                className={`text-xl font-bold mb-6 ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}
              >
                Writing Statistics
              </h3>

              <div className="grid md:grid-cols-2 gap-6">
                <StatCard label="Total Documents" value="24" icon="📄" color="bg-blue-500" darkMode={darkMode} />
                <StatCard label="Words Written" value="12.5K" icon="✍️" color="bg-green-500" darkMode={darkMode} />
                <StatCard label="Voice Sessions" value="18" icon="🎤" color="bg-red-500" darkMode={darkMode} />
                <StatCard label="Days Active" value="45" icon="📅" color="bg-purple-500" darkMode={darkMode} />
              </div>
            </div>

            {/* Account Security */}
            <div
              className={`backdrop-blur-md rounded-2xl p-6 border ${
                darkMode
                  ? 'bg-gray-800/30 border-gray-700/20'
                  : 'bg-white/30 border-white/20'
              }`}
            >
              <h3
                className={`text-xl font-bold mb-6 flex items-center ${
                  darkMode ? 'text-white' : 'text-gray-800'
                }`}
              >
                <KeyIcon className="w-5 h-5 mr-2" />
                Account Security
              </h3>

              <div className="space-y-4">
                <SecurityItem
                  title="Password"
                  subtitle="Last changed 3 months ago"
                  buttonLabel="Change"
                  buttonColor="bg-orange-500 hover:bg-orange-600"
                  darkMode={darkMode}
                />
                <SecurityItem
                  title="Two-Factor Authentication"
                  subtitle="Add an extra layer of security"
                  buttonLabel="Enable"
                  buttonColor="bg-green-500 hover:bg-green-600"
                  darkMode={darkMode}
                />
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

const FormField = ({ label, value, onChange, disabled, multiline = false, darkMode }) => (
  <div>
    <label
      className={`block font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}
    >
      {label}
    </label>
    {multiline ? (
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        rows={3}
        className={`w-full px-4 py-3 rounded-lg border transition-colors ${
          disabled
            ? darkMode
              ? 'bg-gray-700 border-gray-600 text-gray-400'
              : 'bg-gray-100 border-gray-200 text-gray-600'
            : darkMode
            ? 'bg-gray-700 border-gray-600 text-white focus:border-orange-400'
            : 'bg-white/50 border-white/20 text-gray-800 focus:border-orange-300'
        }`}
      />
    ) : (
      <input
        type="text"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        disabled={disabled}
        className={`w-full px-4 py-3 rounded-lg border transition-colors ${
          disabled
            ? darkMode
              ? 'bg-gray-700 border-gray-600 text-gray-400'
              : 'bg-gray-100 border-gray-200 text-gray-600'
            : darkMode
            ? 'bg-gray-700 border-gray-600 text-white focus:border-orange-400'
            : 'bg-white/50 border-white/20 text-gray-800 focus:border-orange-300'
        }`}
      />
    )}
  </div>
);

const StatCard = ({ label, value, icon, color, darkMode }) => (
  <div
    className={`flex items-center p-4 rounded-lg border ${
      darkMode ? 'bg-gray-800/50 border-gray-700/30' : 'bg-white/50 border-white/20'
    }`}
  >
    <div className={`w-12 h-12 ${color} rounded-lg flex items-center justify-center mr-4`}>
      <span className="text-2xl">{icon}</span>
    </div>
    <div>
      <p className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        {value}
      </p>
      <p className={`${darkMode ? 'text-gray-400' : 'text-gray-600'} font-medium`}>
        {label}
      </p>
    </div>
  </div>
);

const SecurityItem = ({ title, subtitle, buttonLabel, buttonColor, darkMode }) => (
  <div
    className={`flex items-center justify-between p-4 rounded-lg border ${
      darkMode ? 'bg-gray-800/50 border-gray-700/30' : 'bg-white/50 border-white/20'
    }`}
  >
    <div>
      <h4 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        {title}
      </h4>
      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>
        {subtitle}
      </p>
    </div>
    <button
      className={`px-4 py-2 text-white rounded-lg transition-colors ${buttonColor}`}
    >
      {buttonLabel}
    </button>
  </div>
);

export default Profile;
