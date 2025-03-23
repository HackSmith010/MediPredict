import React, { useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Menu, X, Sun, Moon } from 'lucide-react';

const Navbar = () => {
  const [isDark, setIsDark] = useState(false);
  const [isOpen, setIsOpen] = useState(false);

  const toggleDarkMode = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <nav className="bg-white dark:bg-gray-800 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          <Link to="/" className="flex items-center space-x-2">
            <Activity className="h-8 w-8 text-[#4A90E2]" />
            <span className="text-xl font-bold text-[#2C3E50] dark:text-white">MediPredict</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center space-x-8">
            <Link to="/" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
              Home
            </Link>
            <Link to="/predict" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
              Predict
            </Link>
            <Link to="/contact" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
              Contact
            </Link>
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition"
            >
              {isDark ? <Sun className="h-5 w-5" /> : <Moon className="h-5 w-5" />}
            </button>
          </div>

          {/* Mobile Navigation */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setIsOpen(!isOpen)}
              className="text-gray-500 hover:text-gray-600 dark:text-gray-400 dark:hover:text-white"
            >
              {isOpen ? <X className="h-6 w-6" /> : <Menu className="h-6 w-6" />}
            </button>
          </div>
        </div>

        {/* Mobile Menu */}
        {isOpen && (
          <div className="md:hidden">
            <div className="px-2 pt-2 pb-3 space-y-1 sm:px-3">
              <Link
                to="/"
                className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700"
              >
                Home
              </Link>
              <Link
                to="/predict"
                className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700"
              >
                Predict
              </Link>
              <Link
                to="/contact"
                className="block px-3 py-2 rounded-md text-base font-medium text-gray-700 hover:text-gray-900 hover:bg-gray-50 dark:text-gray-300 dark:hover:text-white dark:hover:bg-gray-700"
              >
                Contact
              </Link>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
};

export default Navbar;