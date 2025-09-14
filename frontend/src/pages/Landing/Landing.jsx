import React from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { useAuth } from '../../context/AuthContext';
import voiceImg from '../../../assets/7.jpg';
import GestureImg from '../../../assets/2.png';
import macImg from '../../../assets/3.jpg';
import dashwImg from '../../../assets/4.png';
import dashbImg from '../../../assets/5.png';
import { 
  MicrophoneIcon,
  HandRaisedIcon,
  ChevronRightIcon
} from '@heroicons/react/24/outline';

const Landing = () => {
  const { darkMode } = useAuth();

  return (
    <div className={`min-h-screen relative overflow-hidden ${
      darkMode 
        ? 'bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900' 
        : 'bg-gradient-to-br from-orange-100 via-pink-100 to-red-100'
    }`}>
      {/* Animated Background Shapes */}
      <div className="absolute inset-0 overflow-hidden">
        <motion.div
          className={`absolute w-32 h-32 rounded-full opacity-20 ${
            darkMode ? 'bg-purple-500' : 'bg-orange-300'
          }`}
          style={{ top: '10%', left: '10%' }}
          animate={{ y: [0, -20, 0], x: [0, 10, 0] }}
          transition={{ duration: 6, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className={`absolute w-24 h-24 rounded-full opacity-15 ${
            darkMode ? 'bg-pink-500' : 'bg-pink-300'
          }`}
          style={{ top: '60%', right: '15%' }}
          animate={{ y: [0, 15, 0], x: [0, -15, 0] }}
          transition={{ duration: 8, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className={`absolute w-16 h-16 rounded-full opacity-25 ${
            darkMode ? 'bg-orange-500' : 'bg-red-300'
          }`}
          style={{ top: '30%', right: '30%' }}
          animate={{ y: [0, -10, 0], x: [0, 20, 0] }}
          transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className={`absolute w-12 h-12 rounded-full opacity-30 ${
            darkMode ? 'bg-yellow-500' : 'bg-yellow-300'
          }`}
          style={{ bottom: '20%', left: '20%' }}
          animate={{ y: [0, 25, 0], x: [0, -10, 0] }}
          transition={{ duration: 5, repeat: Infinity, ease: "easeInOut" }}
        />
        <motion.div
          className={`absolute w-16 h-16 opacity-20 ${
            darkMode ? 'bg-blue-500' : 'bg-blue-300'
          }`}
          style={{ top: '40%', right: '10%', borderRadius: '4px' }}
          animate={{ y: [0, -15, 0], x: [0, 10, 0], rotate: [0, 45, 0] }}
          transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
        />
      </div>

      {/* Hero Section */}
      <section className="relative py-20 px-4 sm:px-6 lg:px-8 z-10">
        <div className="max-w-4xl mx-auto text-center backdrop-blur-md bg-white/20 rounded-3xl p-12 border border-white/30 shadow-2xl">
          <motion.div 
            className="text-5xl md:text-7xl font-bold mb-6"
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
          >
            {['W','r','i','t','e','M','a','t','e'].map((letter, i) => (
              <motion.span
                key={i}
                className="inline-block bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent"
                initial={{ opacity: 0, y: 50 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: i * 0.1 }}
                whileHover={{ scale: 1.1, transition: { duration: 0.2 } }}
              >
                {letter}
              </motion.span>
            ))}
          </motion.div>
          
          <motion.p 
            className={`text-xl mb-8 max-w-2xl mx-auto ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.2 }}
          >
            An inclusive writing assistant that turns ideas into words for individuals with speech or physical challenges.
          </motion.p>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, delay: 0.4 }}
          >
            <Link
              to="/signup"
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-full text-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105 animate-pulse"
            >
              Get Started Free
              <ChevronRightIcon className="w-5 h-5 ml-2" />
            </Link>
          </motion.div>
        </div>
      </section>

      {/* Core Features */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className={`text-4xl font-bold mb-4 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Core <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">Features</span>
            </h2>
            <p className={`text-xl ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Experience innovative writing technology with voice and gesture controls.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <FeatureCard
              icon={MicrophoneIcon}
              title="Voice Recognition"
              description="Convert your speech to text seamlessly with advanced voice recognition technology."
              color="bg-red-500"
              delay={0.1}
              darkMode={darkMode}
            />
            <FeatureCard
              icon={HandRaisedIcon}
              title="Gesture Control"
              description="Control your writing interface through intuitive hand gestures and movements."
              color="bg-blue-500"
              delay={0.2}
              darkMode={darkMode}
            />
          </div>
        </div>
      </section>

      {/* Project Gallery */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className={`text-4xl font-bold mb-4 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              Project <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">Gallery</span>
            </h2>
            <p className={`text-xl ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              See WriteMate in action through our project screenshots and demonstrations.
            </p>
          </div>

          <div className="grid md:grid-cols-2 gap-8">
            <GalleryCard className="w-full h-48 object-fill rounded-t-xl" title="Voice Recognition Interface" subtitle="Real-time voice to text conversion" image={voiceImg} delay={0.1} darkMode={darkMode}/>
            <GalleryCard title="Gesture Detection System" subtitle="Hand gesture recognition in action" image={GestureImg} delay={0.2} darkMode={darkMode} />
            <GalleryCard title="Writing Interface" subtitle="The WriteMate live editor in action" image={macImg} delay={0.3} darkMode={darkMode} />
            <GalleryCard title="Dashboard Overview" subtitle="Complete project dashboard" image={darkMode ? dashbImg : dashwImg} delay={0.4} darkMode={darkMode} />
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className={`text-4xl font-bold mb-4 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}>
              How <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">It Works</span>
            </h2>
            <p className={`text-xl ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}>
              Simple steps to get started with voice and gesture-controlled writing
            </p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            <StepCard number="1" title="Setup & Connect" description="Sign up and enable voice and gesture controls in your browser settings." color="bg-orange-500" delay={0.1} darkMode={darkMode} />
            <StepCard number="2" title="Start Creating" description="Use voice commands or hand gestures to control your writing interface." color="bg-pink-500" delay={0.2} darkMode={darkMode} />
            <StepCard number="3" title="Save & Share" description="Export your documents and share your hands-free writing experience." color="bg-purple-500" delay={0.3} darkMode={darkMode} />
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20 px-4 sm:px-6 lg:px-8">
        <div className="max-w-4xl mx-auto text-center">
          <motion.h2 
            className={`text-4xl font-bold mb-4 ${
              darkMode ? 'text-white' : 'text-gray-800'
            }`}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
          >
            Ready to Try <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">WriteMate</span>?
          </motion.h2>
          <motion.p 
            className={`text-xl mb-8 ${
              darkMode ? 'text-gray-300' : 'text-gray-600'
            }`}
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
          >
            Experience the future of writing with our innovative voice and gesture control technology.
          </motion.p>
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.4 }}
          >
            <Link
              to="/signup"
              className="inline-flex items-center px-8 py-4 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-full text-lg font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 shadow-lg hover:shadow-xl transform hover:scale-105"
            >
              Get Started Free
              <ChevronRightIcon className="w-5 h-5 ml-2" />
            </Link>
          </motion.div>
        </div>
      </section>
    </div>
  );
};

const FeatureCard = ({ icon: Icon, title, description, color, delay, darkMode }) => (
  <motion.div
    className={`backdrop-blur-md rounded-2xl p-8 border transition-all duration-300 ${
      darkMode
        ? 'bg-gray-800/30 border-gray-700/20 hover:bg-gray-800/40'
        : 'bg-white/30 border-white/20 hover:bg-white/40'
    }`}
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.6, delay }}
    whileHover={{ y: -5, scale: 1.02 }}
  >
    <div className={`w-16 h-16 ${color} rounded-2xl flex items-center justify-center mb-6`}>
      <Icon className="w-8 h-8 text-white" />
    </div>
    <h3 className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
      {title}
    </h3>
    <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{description}</p>
  </motion.div>
);

const GalleryCard = ({ title, subtitle, image, delay, darkMode }) => (
  <motion.div
    className={`backdrop-blur-md rounded-2xl overflow-hidden border transition-all duration-300 ${
      darkMode
        ? 'bg-gray-800/30 border-gray-700/20 hover:bg-gray-800/40'
        : 'bg-white/30 border-white/20 hover:bg-white/40'
    }`}
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.6, delay }}
    whileHover={{ y: -5 }}
  >
    <div className="aspect-video rounded-t-2xl overflow-hidden">
      <img src={image} alt={title} className="w-full h-full object-cover" />
    </div>
    <div className="p-6">
      <h3 className={`text-xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
        {title}
      </h3>
      <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{subtitle}</p>
    </div>
  </motion.div>
);

const StepCard = ({ number, title, description, color, delay, darkMode }) => (
  <motion.div
    className="text-center"
    initial={{ opacity: 0, y: 30 }}
    whileInView={{ opacity: 1, y: 0 }}
    viewport={{ once: true }}
    transition={{ duration: 0.6, delay }}
  >
    <div className={`w-16 h-16 ${color} rounded-full flex items-center justify-center mx-auto mb-6`}>
      <span className="text-2xl font-bold text-white">{number}</span>
    </div>
    <h3 className={`text-2xl font-bold mb-4 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
      {title}
    </h3>
    <p className={`${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{description}</p>
  </motion.div>
);

export default Landing;
