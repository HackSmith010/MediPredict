import React from 'react';
import { Link } from 'react-router-dom';
import { Activity, Mail, Phone } from 'lucide-react';

const Footer = () => {
  return (
    <footer className="bg-white dark:bg-gray-800 shadow-lg mt-auto">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8">
          <div className="col-span-1">
            <div className="flex items-center space-x-2 mb-4">
              <Activity className="h-6 w-6 text-[#4A90E2]" />
              <span className="text-lg font-bold text-[#2C3E50] dark:text-white">MediPredict</span>
            </div>
            <p className="text-[#6C757D] dark:text-gray-300">
              Advanced AI-powered cancer prediction system for early detection and prevention.
            </p>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-[#2C3E50] dark:text-white mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
                  Home
                </Link>
              </li>
              <li>
                <Link to="/predict" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
                  Predict
                </Link>
              </li>
              <li>
                <Link to="/contact" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
                  Contact
                </Link>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-[#2C3E50] dark:text-white mb-4">Contact</h3>
            <ul className="space-y-2">
              <li className="flex items-center space-x-2">
                <Mail className="h-5 w-5 text-[#4A90E2]" />
                <span className="text-[#6C757D] dark:text-gray-300">support@medipredict.com</span>
              </li>
              <li className="flex items-center space-x-2">
                <Phone className="h-5 w-5 text-[#4A90E2]" />
                <span className="text-[#6C757D] dark:text-gray-300">+1 (555) 123-4567</span>
              </li>
            </ul>
          </div>

          <div>
            <h3 className="text-lg font-semibold text-[#2C3E50] dark:text-white mb-4">Legal</h3>
            <ul className="space-y-2">
              <li>
                <Link to="/privacy" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
                  Privacy Policy
                </Link>
              </li>
              <li>
                <Link to="/terms" className="text-[#6C757D] hover:text-[#4A90E2] dark:text-gray-300 dark:hover:text-white transition">
                  Terms & Conditions
                </Link>
              </li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-200 dark:border-gray-700 mt-8 pt-8 text-center">
          <p className="text-[#6C757D] dark:text-gray-300">
            © {new Date().getFullYear()} MediPredict. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;