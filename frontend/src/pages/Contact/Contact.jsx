import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import {
  EnvelopeIcon,
  PhoneIcon,
  MapPinIcon,
  ChatBubbleLeftRightIcon,
  PaperAirplaneIcon
} from '@heroicons/react/24/outline';

const Contact = () => {
  const { darkMode } = useAuth();
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    subject: '',
    message: ''
  });

  const handleSubmit = (e) => {
    e.preventDefault();
    // Handle form submission
    console.log('Form submitted:', formData);
  };

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    });
  };

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
            Contact <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">Us</span>
          </h1>
          <p className={`text-xl max-w-3xl mx-auto ${
            darkMode ? 'text-gray-300' : 'text-gray-600'
          }`}>
            Have questions about WriteMate? We'd love to hear from you. Send us a message and we'll respond as soon as possible.
          </p>
        </motion.div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Contact Form */}
          <motion.div
            className={`backdrop-blur-md rounded-2xl p-8 border ${
              darkMode
                ? 'bg-gray-800/30 border-gray-700/20'
                : 'bg-white/30 border-white/20'
            }`}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.1 }}
          >
            <div className="flex items-center mb-6">
              <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-pink-500 rounded-xl flex items-center justify-center mr-4">
                <ChatBubbleLeftRightIcon className="w-6 h-6 text-white" />
              </div>
              <h2 className={`text-2xl font-bold ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>Send us a Message</h2>
            </div>

            <form onSubmit={handleSubmit} className="space-y-6">
              <div className="grid md:grid-cols-2 gap-6">
                <div>
                  <label className={`block text-sm font-medium mb-2 ${
                    darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    Name
                  </label>
                  <input
                    type="text"
                    name="name"
                    value={formData.name}
                    onChange={handleChange}
                    required
                    className={`w-full px-4 py-3 rounded-lg border outline-none transition-all duration-200 ${
                      darkMode
                        ? 'bg-gray-700/50 border-gray-600/20 text-white focus:bg-gray-700/70 focus:border-orange-400'
                        : 'bg-white/50 border-white/20 text-gray-800 focus:bg-white/70 focus:border-orange-300'
                    }`}
                    placeholder="Your name"
                  />
                </div>
                <div>
                  <label className={`block text-sm font-medium mb-2 ${
                    darkMode ? 'text-gray-300' : 'text-gray-700'
                  }`}>
                    Email
                  </label>
                  <input
                    type="email"
                    name="email"
                    value={formData.email}
                    onChange={handleChange}
                    required
                    className={`w-full px-4 py-3 rounded-lg border outline-none transition-all duration-200 ${
                      darkMode
                        ? 'bg-gray-700/50 border-gray-600/20 text-white focus:bg-gray-700/70 focus:border-orange-400'
                        : 'bg-white/50 border-white/20 text-gray-800 focus:bg-white/70 focus:border-orange-300'
                    }`}
                    placeholder="your@email.com"
                  />
                </div>
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Subject
                </label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleChange}
                  required
                  className={`w-full px-4 py-3 rounded-lg border outline-none transition-all duration-200 ${
                    darkMode
                      ? 'bg-gray-700/50 border-gray-600/20 text-white focus:bg-gray-700/70 focus:border-orange-400'
                      : 'bg-white/50 border-white/20 text-gray-800 focus:bg-white/70 focus:border-orange-300'
                  }`}
                  placeholder="What's this about?"
                />
              </div>

              <div>
                <label className={`block text-sm font-medium mb-2 ${
                  darkMode ? 'text-gray-300' : 'text-gray-700'
                }`}>
                  Message
                </label>
                <textarea
                  name="message"
                  value={formData.message}
                  onChange={handleChange}
                  required
                  rows={6}
                  className={`w-full px-4 py-3 rounded-lg border outline-none transition-all duration-200 resize-none ${
                    darkMode
                      ? 'bg-gray-700/50 border-gray-600/20 text-white focus:bg-gray-700/70 focus:border-orange-400'
                      : 'bg-white/50 border-white/20 text-gray-800 focus:bg-white/70 focus:border-orange-300'
                  }`}
                  placeholder="Tell us more about your question or feedback..."
                />
              </div>

              <button
                type="submit"
                className="w-full flex items-center justify-center px-6 py-3 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
              >
                <PaperAirplaneIcon className="w-5 h-5 mr-2" />
                Send Message
              </button>
            </form>
          </motion.div>

          {/* Contact Information */}
          <motion.div
            className="space-y-8"
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
          >
            {/* Contact Cards */}
            <ContactCard
              icon={EnvelopeIcon}
              title="Email Us"
              content="support@writemate.com"
              description="Send us an email anytime"
              darkMode={darkMode}
              delay={0.3}
            />
            
            <ContactCard
              icon={PhoneIcon}
              title="Call Us"
              content="+1 (555) 123-4567"
              description="Mon-Fri from 8am to 6pm"
              darkMode={darkMode}
              delay={0.4}
            />
            
            <ContactCard
              icon={MapPinIcon}
              title="Visit Us"
              content="123 Innovation Drive"
              description="San Francisco, CA 94105"
              darkMode={darkMode}
              delay={0.5}
            />

            {/* FAQ Section */}
            <motion.div
              className={`backdrop-blur-md rounded-2xl p-6 border ${
                darkMode
                  ? 'bg-gray-800/30 border-gray-700/20'
                  : 'bg-white/30 border-white/20'
              }`}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.6 }}
            >
              <h3 className={`text-xl font-bold mb-4 ${
                darkMode ? 'text-white' : 'text-gray-800'
              }`}>
                Frequently Asked Questions
              </h3>
              <div className="space-y-4">
                <div>
                  <h4 className={`font-semibold mb-2 ${
                    darkMode ? 'text-gray-200' : 'text-gray-700'
                  }`}>
                    How does voice recognition work?
                  </h4>
                  <p className={`text-sm ${
                    darkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>
                    Our advanced AI converts your speech to text in real-time with high accuracy.
                  </p>
                </div>
                <div>
                  <h4 className={`font-semibold mb-2 ${
                    darkMode ? 'text-gray-200' : 'text-gray-700'
                  }`}>
                    Is gesture control supported on all devices?
                  </h4>
                  <p className={`text-sm ${
                    darkMode ? 'text-gray-400' : 'text-gray-600'
                  }`}>
                    Gesture control works on devices with camera access and modern browsers.
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </div>
  );
};

const ContactCard = ({ icon: Icon, title, content, description, darkMode, delay }) => (
  <motion.div
    className={`backdrop-blur-md rounded-2xl p-6 border transition-all duration-300 ${
      darkMode
        ? 'bg-gray-800/30 border-gray-700/20 hover:bg-gray-800/40'
        : 'bg-white/30 border-white/20 hover:bg-white/40'
    }`}
    initial={{ opacity: 0, y: 20 }}
    animate={{ opacity: 1, y: 0 }}
    transition={{ duration: 0.6, delay }}
    whileHover={{ y: -2 }}
  >
    <div className="flex items-start">
      <div className="w-12 h-12 bg-gradient-to-r from-orange-500 to-pink-500 rounded-xl flex items-center justify-center mr-4 flex-shrink-0">
        <Icon className="w-6 h-6 text-white" />
      </div>
      <div>
        <h3 className={`text-lg font-bold mb-1 ${
          darkMode ? 'text-white' : 'text-gray-800'
        }`}>{title}</h3>
        <p className={`font-semibold mb-1 ${
          darkMode ? 'text-gray-200' : 'text-gray-700'
        }`}>{content}</p>
        <p className={`text-sm ${
          darkMode ? 'text-gray-400' : 'text-gray-600'
        }`}>{description}</p>
      </div>
    </div>
  </motion.div>
);

export default Contact;