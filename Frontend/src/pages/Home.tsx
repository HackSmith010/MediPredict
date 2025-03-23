import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { Brain, Shield, Zap } from 'lucide-react';

const Home = () => {
  const navigate = useNavigate();

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex-1"
    >
      
      <div 
        className="relative bg-gradient-to-r from-[#4A90E2] to-[#008080] py-24"
        style={{
          backgroundImage: `linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)), url('https://images.unsplash.com/photo-1576091160550-2173dba999ef?auto=format&fit=crop&q=80')`,
          backgroundSize: 'cover',
          backgroundPosition: 'center',
        }}
      >
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <motion.h1 
            className="text-4xl md:text-6xl font-bold text-white mb-6"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.2 }}
          >
            AI-powered Cancer Prediction System
          </motion.h1>
          <motion.p 
            className="text-xl text-gray-200 mb-8 max-w-2xl mx-auto"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.4 }}
          >
            Advanced machine learning algorithms to help detect early signs of cancer
          </motion.p>
          <motion.button
            onClick={() => navigate('/predict')}
            className="bg-white text-[#4A90E2] px-8 py-3 rounded-full font-semibold text-lg hover:bg-gray-100 transition duration-300 transform hover:scale-105"
            initial={{ y: 20, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 0.6 }}
          >
            Start Prediction
          </motion.button>
        </div>
      </div>

      
      <div className="py-16 bg-white dark:bg-gray-800">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-[#2C3E50] dark:text-white mb-4">Why Choose Us?</h2>
            <p className="text-[#6C757D] dark:text-gray-300">State-of-the-art technology combined with medical expertise</p>
          </div>

          <div className="grid md:grid-cols-3 gap-8">
            {[
              {
                icon: <Brain className="h-12 w-12 text-[#4A90E2]" />,
                title: "AI-Powered",
                description: "Advanced machine learning algorithms for accurate predictions"
              },
              {
                icon: <Shield className="h-12 w-12 text-[#008080]" />,
                title: "Secure",
                description: "Your medical data is protected with enterprise-grade security"
              },
              {
                icon: <Zap className="h-12 w-12 text-[#4A90E2]" />,
                title: "Fast Results",
                description: "Get instant predictions and recommendations"
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                className="bg-gray-50 dark:bg-gray-700 p-6 rounded-xl text-center"
                initial={{ y: 20, opacity: 0 }}
                animate={{ y: 0, opacity: 1 }}
                transition={{ delay: 0.2 * index }}
              >
                <div className="flex justify-center mb-4">{feature.icon}</div>
                <h3 className="text-xl font-semibold text-[#2C3E50] dark:text-white mb-2">{feature.title}</h3>
                <p className="text-[#6C757D] dark:text-gray-300">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </motion.div>
  );
};

export default Home;