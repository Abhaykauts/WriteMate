import React from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import {
  SparklesIcon,
  MicrophoneIcon,
  HandRaisedIcon,
  UserGroupIcon,
  LightBulbIcon,
  HeartIcon
} from '@heroicons/react/24/outline';

const About = () => {
  const { darkMode } = useAuth();

  return (
    <div className="min-h-screen pt-24 pb-8 px-4 sm:px-6 lg:px-8">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <h1 className={`text-5xl font-bold mb-6 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            About <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">WriteMate</span>
          </h1>
          <p className={`text-xl max-w-3xl mx-auto ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Revolutionizing the way people write with cutting-edge AI technology, voice recognition, and gesture control for a truly hands-free writing experience.
          </p>
        </motion.div>

        {/* Mission Section */}
        <motion.div
          className={`backdrop-blur-md rounded-2xl p-8 border mb-12 ${
            darkMode
              ? 'bg-gray-800/30 border-gray-700/20'
              : 'bg-white/30 border-white/20'
          }`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="text-center">
            <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-6">
              <SparklesIcon className="w-8 h-8 text-white" />
            </div>
            <h2 className={`text-3xl font-bold mb-4 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>Our Mission</h2>
            <p className={`text-lg leading-relaxed max-w-4xl mx-auto ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              At WriteMate, we believe that everyone should have access to powerful writing tools that adapt to their needs. 
              Our mission is to break down barriers in digital writing by creating an inclusive platform that supports 
              voice recognition, gesture control, and AI-powered assistance, making writing accessible to everyone regardless 
              of their physical abilities or technical expertise.
            </p>
          </div>
        </motion.div>

        {/* Features Grid */}
        <div className="grid md:grid-cols-3 gap-8 mb-16">
          <FeatureCard
            icon={MicrophoneIcon}
            title="Voice-First Design"
            description="Advanced speech recognition technology that understands natural language and converts your thoughts into text seamlessly."
            color="bg-red-500"
            delay={0.2}
            darkMode={darkMode}
          />
          <FeatureCard
            icon={HandRaisedIcon}
            title="Gesture Control"
            description="Innovative hand gesture recognition that allows you to control your writing interface without touching a keyboard or mouse."
            color="bg-blue-500"
            delay={0.3}
            darkMode={darkMode}
          />
          <FeatureCard
            icon={SparklesIcon}
            title="AI-Powered"
            description="Smart writing assistance that helps improve your content, suggests improvements, and learns from your writing style."
            color="bg-purple-500"
            delay={0.4}
            darkMode={darkMode}
          />
        </div>

        {/* Team Section */}
        <motion.div
          className="text-center mb-16"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <h2 className={`text-4xl font-bold mb-8 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Built with <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">Passion</span>
          </h2>
          
          <div className="grid md:grid-cols-3 gap-8">
            <ValueCard
              icon={UserGroupIcon}
              title="Inclusive Design"
              description="We design for everyone, ensuring our platform is accessible to users with different abilities and needs."
              darkMode={darkMode}
              delay={0.6}
            />
            <ValueCard
              icon={LightBulbIcon}
              title="Innovation"
              description="Constantly pushing the boundaries of what's possible in writing technology and user experience."
              darkMode={darkMode}
              delay={0.7}
            />
            <ValueCard
              icon={HeartIcon}
              title="User-Centric"
              description="Every feature we build is designed with our users in mind, focusing on real needs and genuine value."
              darkMode={darkMode}
              delay={0.8}
            />
          </div>
        </motion.div>

        {/* CTA Section */}
        <motion.div
          className={`backdrop-blur-md rounded-2xl p-8 border text-center ${
            darkMode
              ? 'bg-gray-800/30 border-gray-700/20'
              : 'bg-white/30 border-white/20'
          }`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.9 }}
        >
          <h3 className={`text-3xl font-bold mb-4 ${
            darkMode ? 'text-white' : 'text-gray-800'
          }`}>
            Ready to Transform Your Writing?
          </h3>
          <p className={`text-lg mb-6 ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Join thousands of writers who have already discovered the power of hands-free writing.
          </p>
          <button className="px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-full text-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105">
            Get Started Today
          </button>
        </motion.div>
      </div>
    </div>
  );
};

const FeatureCard = ({ icon: Icon, title, description, color, delay, darkMode }) => (
  <motion.div
    className={`backdrop-blur-md rounded-2xl p-6 border transition-all duration-300 ${
      darkMode
        ? 'bg-gray-800/30 border-gray-700/20 hover:bg-gray-800/40'
        : 'bg-white/30 border-white/20 hover:bg-white/40'
    }`}
    initial={{ opacity: 0, y: 30 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
    whileHover={{ y: -5 }}
  >
    <div className={`w-12 h-12 ${color} rounded-xl flex items-center justify-center mb-4`}>
      <Icon className="w-6 h-6 text-white" />
    </div>
    <h3 className={`text-xl font-bold mb-3 ${
      darkMode ? 'text-white' : 'text-gray-800'
    }`}>{title}</h3>
    <p className={`leading-relaxed ${
      darkMode ? 'text-gray-300' : 'text-gray-600'
    }`}>{description}</p>
  </motion.div>
);

const ValueCard = ({ icon: Icon, title, description, darkMode, delay }) => (
  <motion.div
    className="text-center"
    initial={{ opacity: 0, y: 30 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
  >
    <div className="w-16 h-16 bg-gradient-to-r from-orange-500 to-pink-500 rounded-2xl flex items-center justify-center mx-auto mb-4">
      <Icon className="w-8 h-8 text-white" />
    </div>
    <h3 className={`text-xl font-bold mb-3 ${
      darkMode ? 'text-white' : 'text-gray-800'
    }`}>{title}</h3>
    <p className={`leading-relaxed ${
      darkMode ? 'text-gray-300' : 'text-gray-600'
    }`}>{description}</p>
  </motion.div>
);

export default About;