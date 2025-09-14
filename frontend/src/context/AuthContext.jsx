import React, { createContext, useContext, useState, useEffect } from 'react';

const AuthContext = createContext();

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);
  const [voiceEnabled, setVoiceEnabled] = useState(false);
  const [gestureEnabled, setGestureEnabled] = useState(false);
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Simulate auth check
    const checkAuth = async () => {
      const savedUser = localStorage.getItem('writemate_user');
      if (savedUser) {
        setUser(JSON.parse(savedUser));
      }
      setLoading(false);
    };
    
    checkAuth();
  }, []);

  const toggleDarkMode = () => {
    setDarkMode(!darkMode);
  };

  const signIn = async (email, password) => {
    // Simulate sign in
    const userData = {
      id: '1',
      name: 'Akash singh',
      email: email,
      avatar: 'A'
    };
    localStorage.setItem('writemate_user', JSON.stringify(userData));
    setUser(userData);
    return userData;
  };

  const signUp = async (name, email, password) => {
    // Simulate sign up
    const userData = {
      id: '1',
      name: name,
      email: email,
      avatar: name.charAt(0).toUpperCase()
    };
    localStorage.setItem('writemate_user', JSON.stringify(userData));
    setUser(userData);
    return userData;
  };

  const signOut = () => {
    localStorage.removeItem('writemate_user');
    setUser(null);
  };

  const value = {
    user,
    signIn,
    signUp,
    signOut,
    loading,
    voiceEnabled,
    setVoiceEnabled,
    gestureEnabled,
    setGestureEnabled,
    darkMode,
    setDarkMode,
    toggleDarkMode
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};