import React, { useState } from 'react';
import { motion } from 'framer-motion';
import {
  MagnifyingGlassIcon,
  PlusIcon,
  EyeIcon,
  ShareIcon,
  PencilSquareIcon,
  ArrowDownTrayIcon,
  TrashIcon
} from '@heroicons/react/24/outline';
import { useAuth } from '../../context/AuthContext';

const Documents = () => {
  const { darkMode } = useAuth();
  const [searchTerm, setSearchTerm] = useState('');
  const [activeFilter, setActiveFilter] = useState('All');

  const documents = [
    {
      id: 1,
      title: 'Project Proposal Draft',
      type: 'Document',
      status: 'draft',
      words: 1250,
      modified: '2 hours ago'
    },
    {
      id: 2,
      title: 'Meeting Notes - Q4 Planning',
      type: 'Notes',
      status: 'complete',
      words: 850,
      modified: '1 day ago'
    },
    {
      id: 3,
      title: 'Blog Post: WriteMate Review',
      type: 'Blog',
      status: 'review',
      words: 2100,
      modified: '3 days ago'
    },
    {
      id: 4,
      title: 'User Research Summary',
      type: 'Report',
      status: 'complete',
      words: 3200,
      modified: '1 week ago'
    },
    {
      id: 5,
      title: 'Product Roadmap 2024',
      type: 'Document',
      status: 'draft',
      words: 1800,
      modified: '2 weeks ago'
    }
  ];

  const filters = ['All', 'Documents', 'Notes', 'Blogs'];
  const stats = [
    { label: 'Total Documents', value: '5', color: 'text-blue-600' },
    { label: 'Drafts', value: '2', color: 'text-orange-600' },
    { label: 'Total Words', value: '9,200', color: 'text-green-600' },
    { label: 'Completed', value: '2', color: 'text-purple-600' }
  ];

  return (
    <div className={`min-h-screen pt-24 pb-8 px-4 sm:px-6 lg:px-8 ${darkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div
          className="flex flex-col md:flex-row md:items-center justify-between mb-8"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div>
            <h1 className={`text-4xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>
              My <span className="bg-gradient-to-r from-orange-500 to-pink-500 bg-clip-text text-transparent">Documents</span>
            </h1>
            <p className={`text-xl ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
              Manage and organize your writing projects
            </p>
          </div>
          <button className="mt-4 md:mt-0 px-6 py-3 bg-gradient-to-r from-orange-500 to-pink-500 text-white rounded-full font-semibold hover:from-orange-600 hover:to-pink-600 transition-all duration-200 flex items-center space-x-2 shadow-lg">
            <PlusIcon className="w-5 h-5" />
            <span>New Document</span>
          </button>
        </motion.div>

        {/* Search and Filters */}
        <motion.div
          className={`backdrop-blur-md rounded-2xl p-6 border border-white/20 mb-8 ${
            darkMode ? 'bg-gray-800/50' : 'bg-white/30'
          }`}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
        >
          <div className="flex flex-col md:flex-row md:items-center justify-between space-y-4 md:space-y-0">
            {/* Search */}
            <div className="relative flex-1 md:mr-6">
              <MagnifyingGlassIcon
                className={`w-5 h-5 absolute left-3 top-1/2 transform -translate-y-1/2 ${
                  darkMode ? 'text-gray-400' : 'text-gray-500'
                }`}
              />
              <input
                type="text"
                placeholder="Search documents..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className={`w-full pl-10 pr-4 py-3 rounded-xl border border-white/20 outline-none transition-all duration-200 ${
                  darkMode
                    ? 'bg-gray-700 text-white placeholder-gray-400 focus:bg-gray-600 focus:border-orange-400'
                    : 'bg-white/50 text-gray-800 placeholder-gray-400 focus:bg-white/70 focus:border-orange-300'
                }`}
              />
            </div>

            {/* Filters */}
            <div className="flex space-x-2">
              {filters.map((filter) => (
                <button
                  key={filter}
                  onClick={() => setActiveFilter(filter)}
                  className={`px-4 py-2 rounded-lg font-medium transition-all duration-200 ${
                    activeFilter === filter
                      ? 'bg-orange-500 text-white'
                      : darkMode
                      ? 'bg-gray-700 text-gray-200 hover:bg-gray-600'
                      : 'bg-white/50 text-gray-700 hover:bg-white/70'
                  }`}
                >
                  {filter}
                </button>
              ))}
            </div>
          </div>
        </motion.div>

        {/* Documents Grid */}
        <div className="grid md:grid-cols-2 xl:grid-cols-3 gap-6 mb-12">
          {documents.map((doc, index) => (
            <DocumentCard key={doc.id} document={doc} delay={index * 0.1} darkMode={darkMode} />
          ))}
        </div>

        {/* Stats */}
        <motion.div
          className="grid md:grid-cols-4 gap-6"
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.4 }}
        >
          {stats.map((stat, index) => (
            <div key={index} className="text-center">
              <div className={`text-4xl font-bold ${stat.color} mb-2`}>
                {stat.value}
              </div>
              <div className={`${darkMode ? 'text-gray-300' : 'text-gray-600'} font-medium`}>
                {stat.label}
              </div>
            </div>
          ))}
        </motion.div>
      </div>
    </div>
  );
};

const DocumentCard = ({ document, delay, darkMode }) => {
  const getStatusColor = (status) => {
    switch (status) {
      case 'complete': return 'bg-green-100 text-green-800';
      case 'draft': return 'bg-orange-100 text-orange-800';
      case 'review': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusText = (status) => {
    switch (status) {
      case 'complete': return 'Complete';
      case 'draft': return 'Draft';
      case 'review': return 'Review';
      default: return 'Unknown';
    }
  };

  const getTypeIcon = (type) => {
    switch (type) {
      case 'Document': return '📄';
      case 'Notes': return '📝';
      case 'Blog': return '📖';
      case 'Report': return '📊';
      default: return '📄';
    }
  };

  return (
    <motion.div
      className={`backdrop-blur-md rounded-2xl p-6 border border-white/20 transition-all duration-300 hover:shadow-lg ${
        darkMode ? 'bg-gray-800/50 hover:bg-gray-700' : 'bg-white/30 hover:bg-white/40'
      }`}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.6, delay }}
      whileHover={{ y: -5 }}
    >
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div className="flex items-center space-x-2">
          <span className="text-2xl">{getTypeIcon(document.type)}</span>
          <span className={`px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(document.status)}`}>
            {getStatusText(document.status)}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          <button className={`p-1 transition-colors ${darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-400 hover:text-gray-600'}`}>
            <EyeIcon className="w-4 h-4" />
          </button>
          <button className={`p-1 transition-colors ${darkMode ? 'text-gray-400 hover:text-gray-200' : 'text-gray-400 hover:text-gray-600'}`}>
            <ShareIcon className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Content */}
      <h3 className={`text-xl font-bold mb-2 ${darkMode ? 'text-white' : 'text-gray-800'}`}>{document.title}</h3>
      <p className={`text-sm mb-4 ${darkMode ? 'text-gray-400' : 'text-gray-600'}`}>{document.type}</p>
      
      <div className={`flex items-center justify-between text-sm mb-6 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
        <span>{document.words} words</span>
        <span>{document.modified}</span>
      </div>

      {/* Actions */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <button className={`p-2 rounded-lg transition-colors ${darkMode ? 'text-gray-300 hover:text-orange-400 hover:bg-gray-700' : 'text-gray-600 hover:text-orange-600 hover:bg-orange-50'}`}>
            <PencilSquareIcon className="w-4 h-4" />
          </button>
          <button className={`p-2 rounded-lg transition-colors ${darkMode ? 'text-gray-300 hover:text-blue-400 hover:bg-gray-700' : 'text-gray-600 hover:text-blue-600 hover:bg-blue-50'}`}>
            <ArrowDownTrayIcon className="w-4 h-4" />
          </button>
          <button className={`p-2 rounded-lg transition-colors ${darkMode ? 'text-gray-300 hover:text-red-400 hover:bg-gray-700' : 'text-gray-600 hover:text-red-600 hover:bg-red-50'}`}>
            <TrashIcon className="w-4 h-4" />
          </button>
        </div>
        <button className="px-4 py-2 bg-orange-500 text-white rounded-lg hover:bg-orange-600 transition-colors font-medium">
          Open
        </button>
      </div>
    </motion.div>
  );
};

export default Documents;
