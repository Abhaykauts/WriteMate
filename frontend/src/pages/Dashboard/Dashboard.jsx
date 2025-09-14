import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import {
  PlusIcon,
  MicrophoneIcon,
  HandRaisedIcon,
  DocumentTextIcon,
  PencilIcon,
  SpeakerWaveIcon,
  UsersIcon,
  ClockIcon,
  PencilSquareIcon
} from '@heroicons/react/24/outline';

const Dashboard = () => {
  const { user, darkMode } = useAuth();

  return (
    <div className={`min-h-screen pt-24 pb-8 px-4 sm:px-6 lg:px-8 transition-colors duration-300 ${
      darkMode ? 'bg-gray-900' : 'bg-gray-50'
    }`}>
      <div className="max-w-7xl mx-auto">
        {/* Welcome Section */}
        <motion.div
          className="mb-12"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className={`text-4xl font-bold mb-2 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Welcome back,{" "}
            <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">
              {user?.name || 'User'}!
            </span>
          </h1>
          <p className={`text-xl ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Ready to create something amazing today?
          </p>
        </motion.div>

        {/* Quick Actions */}
        <div className="grid md:grid-cols-3 gap-6 mb-12">
          <QuickActionCard
            to="/write"
            icon={PlusIcon}
            title="New Document"
            description="Start writing with voice, gestures, or keyboard"
            color="bg-orange-500"
            delay={0.1}
            darkMode={darkMode}
          />
          <QuickActionCard
            to="/write"
            icon={MicrophoneIcon}
            title="Voice Note"
            description="Quickly capture ideas with voice recording"
            color="bg-red-500"
            delay={0.2}
            darkMode={darkMode}
          />
          <QuickActionCard
            to="/write"
            icon={HandRaisedIcon}
            title="Gesture Mode"
            description="Experience hands-free writing"
            color="bg-blue-500"
            delay={0.3}
            darkMode={darkMode}
          />
        </div>

        {/* Stats Grid */}
        <div className="grid md:grid-cols-4 gap-6 mb-12">
          <StatCard
            icon={DocumentTextIcon}
            label="Documents"
            value="24"
            color="bg-blue-500"
            delay={0.1}
            darkMode={darkMode}
          />
          <StatCard
            icon={PencilIcon}
            label="Words Written"
            value="12.5K"
            color="bg-green-500"
            delay={0.2}
            darkMode={darkMode}
          />
          <StatCard
            icon={SpeakerWaveIcon}
            label="Voice Sessions"
            value="18"
            color="bg-red-500"
            delay={0.3}
            darkMode={darkMode}
          />
          <StatCard
            icon={UsersIcon}
            label="Collaborators"
            value="6"
            color="bg-purple-500"
            delay={0.4}
            darkMode={darkMode}
          />
        </div>

        <div className="grid lg:grid-cols-2 gap-8">
          {/* Recent Documents */}
          <motion.div
            className={`backdrop-blur-md rounded-2xl p-6 border transition-colors duration-300 ${
              darkMode
                ? 'bg-gray-800/50 border-gray-700'
                : 'bg-white/30 border-white/20'
            }`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.5 }}
          >
            <div className="flex items-center justify-between mb-6">
              <h2 className={`text-2xl font-bold flex items-center ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                <DocumentTextIcon className="w-6 h-6 mr-2" />
                Recent Documents
              </h2>
              <Link
                to="/documents"
                className="text-orange-500 hover:text-orange-600 font-medium"
              >
                View All
              </Link>
            </div>

            <div className="space-y-4">
              <DocumentItem
                title="Project Proposal"
                type="Document"
                time="2 hours ago"
                status="draft"
                darkMode={darkMode}
              />
              <DocumentItem
                title="Meeting Notes"
                type="Notes"
                time="1 day ago"
                status="complete"
                darkMode={darkMode}
              />
              <DocumentItem
                title="Blog Draft"
                type="Blog"
                time="3 days ago"
                status="draft"
                darkMode={darkMode}
              />
            </div>
          </motion.div>

          {/* Writing Progress & Schedule */}
          <div className="space-y-8">
            {/* Writing Progress */}
            <motion.div
              className={`backdrop-blur-md rounded-2xl p-6 border transition-colors duration-300 ${
                darkMode
                  ? 'bg-gray-800/50 border-gray-700'
                  : 'bg-white/30 border-white/20'
              }`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <h2 className={`text-2xl font-bold mb-6 flex items-center ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                <PencilSquareIcon className="w-6 h-6 mr-2" />
                Writing Progress
              </h2>

              <div className="space-y-4">
                <ProgressItem
                  label="Daily Goal"
                  percentage={75}
                  color="bg-orange-500"
                  darkMode={darkMode}
                />
                <ProgressItem
                  label="Weekly Goal"
                  percentage={60}
                  color="bg-blue-500"
                  darkMode={darkMode}
                />
              </div>
            </motion.div>

            {/* Today's Schedule */}
            <motion.div
              className={`backdrop-blur-md rounded-2xl p-6 border transition-colors duration-300 ${
                darkMode
                  ? 'bg-gray-800/50 border-gray-700'
                  : 'bg-white/30 border-white/20'
              }`}
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ duration: 0.6, delay: 0.7 }}
            >
              <h2 className={`text-2xl font-bold mb-6 flex items-center ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                <ClockIcon className="w-6 h-6 mr-2" />
                Today's Schedule
              </h2>

              <div className="space-y-4">
                <ScheduleItem
                  title="Team meeting"
                  time="10:00 AM"
                  color="bg-orange-400"
                  darkMode={darkMode}
                />
                <ScheduleItem
                  title="Writing session"
                  time="2:00 PM"
                  color="bg-blue-400"
                  darkMode={darkMode}
                />
                <ScheduleItem
                  title="Review deadline"
                  time="5:00 PM"
                  color="bg-green-400"
                  darkMode={darkMode}
                />
              </div>
            </motion.div>
          </div>
        </div>
      </div>
    </div>
  );
};

/* ---------- Sub Components ---------- */

const QuickActionCard = ({ to, icon: Icon, title, description, color, delay, darkMode }) => (
  <motion.div
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
    whileHover={{ y: -5 }}
  >
    <Link
      to={to}
      className={`block backdrop-blur-md rounded-2xl p-6 border transition-all duration-300 group ${
        darkMode ? 'bg-gray-800/50 border-gray-700 hover:bg-gray-700/50' : 'bg-white/30 border-white/20 hover:bg-white/40'
      }`}
    >
      <div className={`w-16 h-16 ${color} rounded-2xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
        <Icon className="w-8 h-8 text-white" />
      </div>
      <h3 className={`text-xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        {title}
      </h3>
      <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
        {description}
      </p>
    </Link>
  </motion.div>
);

const StatCard = ({ icon: Icon, label, value, color, delay, darkMode }) => (
  <motion.div
    className={`backdrop-blur-md rounded-2xl p-6 border transition-colors duration-300 ${
      darkMode ? 'bg-gray-800/50 border-gray-700' : 'bg-white/30 border-white/20'
    }`}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
    whileHover={{ scale: 1.05 }}
  >
    <div className="flex items-center justify-between">
      <div>
        <p className={`text-sm font-medium mb-1 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{label}</p>
        <p className={`text-3xl font-bold ${darkMode ? 'text-white' : 'text-gray-800'}`}>{value}</p>
      </div>
      <div className={`w-12 h-12 ${color} rounded-xl flex items-center justify-center`}>
        <Icon className="w-6 h-6 text-white" />
      </div>
    </div>
  </motion.div>
);

const DocumentItem = ({ title, type, time, status, darkMode }) => (
  <div className={`flex items-center p-4 rounded-xl transition-colors cursor-pointer ${
    darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-white/30'
  }`}>
    <div className="w-10 h-10 bg-orange-100 rounded-lg flex items-center justify-center mr-4">
      <DocumentTextIcon className="w-5 h-5 text-orange-600" />
    </div>
    <div className="flex-1">
      <h4 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>{title}</h4>
      <p className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{type} • {time}</p>
    </div>
    <div className="flex items-center space-x-2">
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${
        status === 'complete'
          ? 'bg-green-100 text-green-800'
          : 'bg-orange-100 text-orange-800'
      }`}>
        {status === 'complete' ? 'Complete' : 'Draft'}
      </span>
      <PencilSquareIcon className="w-4 h-4 text-gray-400" />
    </div>
  </div>
);

const ProgressItem = ({ label, percentage, color, darkMode }) => (
  <div>
    <div className="flex justify-between items-center mb-2">
      <span className={`${darkMode ? 'text-gray-300' : 'text-gray-700'} font-medium`}>{label}</span>
      <span className={`${darkMode ? 'text-white' : 'text-gray-700'} font-bold`}>{percentage}%</span>
    </div>
    <div className="w-full bg-gray-200 rounded-full h-2">
      <div
        className={`${color} h-2 rounded-full transition-all duration-300`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  </div>
);

const ScheduleItem = ({ title, time, color, darkMode }) => (
  <div className={`flex items-center p-3 rounded-lg transition-colors ${
    darkMode ? 'hover:bg-gray-700/50' : 'hover:bg-white/30'
  }`}>
    <div className={`w-3 h-3 ${color} rounded-full mr-3`} />
    <div className="flex-1">
      <h4 className={`font-semibold ${darkMode ? 'text-white' : 'text-gray-800'}`}>{title}</h4>
    </div>
    <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{time}</span>
  </div>
);

export default Dashboard;
