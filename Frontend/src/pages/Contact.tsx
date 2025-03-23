import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Mail, Phone, MapPin, Send } from 'lucide-react';

const Contact = () => {
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    message: '',
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle form submission
    console.log('Form submitted:', formData);
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex-1 py-12 bg-[#F5F7FA] dark:bg-gray-900"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 gap-8">
          <div>
            <h1 className="text-3xl font-bold text-[#2C3E50] dark:text-white mb-6">
              Get in Touch
            </h1>
            <p className="text-[#6C757D] dark:text-gray-300 mb-8">
              Have questions about our cancer prediction system? We're here to help.
              Contact our team for support or more information.
            </p>

            <div className="space-y-6">
              <div className="flex items-center space-x-4">
                <Mail className="h-6 w-6 text-[#4A90E2]" />
                <div>
                  <h3 className="text-lg font-semibold text-[#2C3E50] dark:text-white">
                    Email
                  </h3>
                  <p className="text-[#6C757D] dark:text-gray-300">
                    support@medipredict.com
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <Phone className="h-6 w-6 text-[#4A90E2]" />
                <div>
                  <h3 className="text-lg font-semibold text-[#2C3E50] dark:text-white">
                    Phone
                  </h3>
                  <p className="text-[#6C757D] dark:text-gray-300">
                    +1 (555) 123-4567
                  </p>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <MapPin className="h-6 w-6 text-[#4A90E2]" />
                <div>
                  <h3 className="text-lg font-semibold text-[#2C3E50] dark:text-white">
                    Location
                  </h3>
                  <p className="text-[#6C757D] dark:text-gray-300">
                    123 Medical Center Dr, Healthcare City, HC 12345
                  </p>
                </div>
              </div>
            </div>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
            <form onSubmit={handleSubmit} className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                  Name
                </label>
                <input
                  type="text"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                  Email
                </label>
                <input
                  type="email"
                  value={formData.email}
                  onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                  Message
                </label>
                <textarea
                  value={formData.message}
                  onChange={(e) => setFormData({ ...formData, message: e.target.value })}
                  rows={4}
                  className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                  required
                />
              </div>

              <button
                type="submit"
                className="w-full bg-[#4A90E2] hover:bg-[#357ABD] text-white font-semibold py-3 px-6 rounded-lg transition duration-300 flex items-center justify-center space-x-2"
              >
                <Send className="h-5 w-5" />
                <span>Send Message</span>
              </button>
            </form>
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Contact;