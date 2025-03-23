import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { AlertCircle } from 'lucide-react';
import axios from 'axios';

// Define types for our prediction results
interface PredictionResult {
  prediction: string;
  probability: number;
  status: string;
}

interface CancerFeatures {
  model: string;
  features: string[];
  status: string;
}

const Predict = () => {
  const navigate = useNavigate();
  const [cancerType, setCancerType] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [breastFeatures, setBreastFeatures] = useState<string[]>([]);
  const [lungFeatures, setLungFeatures] = useState<string[]>([]);
  
  // For breast cancer form
  const [radiusMean, setRadiusMean] = useState('');
  const [textureMean, setTextureMean] = useState('');
  const [perimeterMean, setPerimeterMean] = useState('');
  const [areaMean, setAreaMean] = useState('');
  const [smoothnessMean, setSmoothnessMean] = useState('');
  const [compactnessMean, setCompactnessMean] = useState('');
  const [concavityMean, setConcavityMean] = useState('');
  const [concavePointsMean, setConcavePointsMean] = useState('');
  const [symmetryMean, setSymmetryMean] = useState('');
  const [fractalDimensionMean, setFractalDimensionMean] = useState('');
  
  // For lung cancer form
  const [age, setAge] = useState('');
  const [alcoholConsuming, setAlcoholConsuming] = useState('0');
  const [allergy, setAllergy] = useState('0');
  const [peerPressure, setPeerPressure] = useState('0');
  const [yellowFingers, setYellowFingers] = useState('0');
  const [fatigue, setFatigue] = useState('0');
  const [coughing, setCoughing] = useState('0');

  // API endpoint
  const API_BASE_URL = 'http://localhost:5000/api';

  // Fetch features from the backend when component mounts
  useEffect(() => {
    const fetchFeatures = async () => {
      try {
        // Fetch breast cancer features
        const breastResponse = await axios.get<CancerFeatures>(`${API_BASE_URL}/info/breast`);
        if (breastResponse.data.status === 'success') {
          setBreastFeatures(breastResponse.data.features);
        }
        
        // Fetch lung cancer features
        const lungResponse = await axios.get<CancerFeatures>(`${API_BASE_URL}/info/lung`);
        if (lungResponse.data.status === 'success') {
          setLungFeatures(lungResponse.data.features);
        }
      } catch (err) {
        console.error('Error fetching features:', err);
      }
    };
    
    fetchFeatures();
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      let response;
      
      if (cancerType === 'breast') {
        // Prepare breast cancer data
        const breastCancerData = {
          radius_mean: parseFloat(radiusMean),
          texture_mean: parseFloat(textureMean),
          perimeter_mean: parseFloat(perimeterMean),
          area_mean: parseFloat(areaMean),
          smoothness_mean: parseFloat(smoothnessMean),
          compactness_mean: parseFloat(compactnessMean),
          concavity_mean: parseFloat(concavityMean),
          concave_points_mean: parseFloat(concavePointsMean),
          symmetry_mean: parseFloat(symmetryMean),
          fractal_dimension_mean: parseFloat(fractalDimensionMean)
        };
        
        // Call breast cancer API
        response = await axios.post<PredictionResult>(
          `${API_BASE_URL}/predict/breast`, 
          breastCancerData
        );
      } else if (cancerType === 'lung') {
        // Prepare lung cancer data according to the actual backend features
        // Making sure to match the expected parameter names
        const lungCancerData = {
          age: parseFloat(age),
          alcohol_consuming: parseFloat(alcoholConsuming),
          allergy_: parseFloat(allergy),
          peer_pressure: parseFloat(peerPressure),
          yellow_fingers: parseFloat(yellowFingers),
          fatigue_: parseFloat(fatigue),
          coughing: parseFloat(coughing)
        };
        
        // Call lung cancer API
        response = await axios.post<PredictionResult>(
          `${API_BASE_URL}/predict/lung`, 
          lungCancerData
        );
      } else {
        throw new Error('Please select a cancer type');
      }
      
      // Store results in localStorage to access on results page
      localStorage.setItem('predictionResult', JSON.stringify(response.data));
      
      // Store the cancer type in localStorage
      localStorage.setItem('cancerType', cancerType);
      
      // Navigate to results page
      navigate('/results');
    } catch (err) {
      console.error('Prediction error:', err);
      setError(err instanceof Error ? err.message : 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex-1 py-12 bg-[#F5F7FA] dark:bg-gray-900"
    >
      <div className="max-w-3xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8">
          <h1 className="text-3xl font-bold text-[#2C3E50] dark:text-white mb-6 text-center">
            Cancer Prediction
          </h1>

          {error && (
            <div className="mb-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded flex items-center">
              <AlertCircle className="mr-2" size={18} />
              <span>{error}</span>
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-6">
            <div>
              <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                Select Cancer Type
              </label>
              <select
                value={cancerType}
                onChange={(e) => setCancerType(e.target.value)}
                className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                required
              >
                <option value="">Select type...</option>
                <option value="breast">Breast Cancer</option>
                <option value="lung">Lung Cancer</option>
              </select>
            </div>

            {cancerType === 'breast' && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Radius Mean
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={radiusMean}
                      onChange={(e) => setRadiusMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Texture Mean
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={textureMean}
                      onChange={(e) => setTextureMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Perimeter Mean
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={perimeterMean}
                      onChange={(e) => setPerimeterMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Area Mean
                    </label>
                    <input
                      type="number"
                      step="0.01"
                      value={areaMean}
                      onChange={(e) => setAreaMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Smoothness Mean
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={smoothnessMean}
                      onChange={(e) => setSmoothnessMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Compactness Mean
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={compactnessMean}
                      onChange={(e) => setCompactnessMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Concavity Mean
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={concavityMean}
                      onChange={(e) => setConcavityMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Concave Points Mean
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={concavePointsMean}
                      onChange={(e) => setConcavePointsMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Symmetry Mean
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={symmetryMean}
                      onChange={(e) => setSymmetryMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Fractal Dimension Mean
                    </label>
                    <input
                      type="number"
                      step="0.001"
                      value={fractalDimensionMean}
                      onChange={(e) => setFractalDimensionMean(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                </div>
              </div>
            )}

            {cancerType === 'lung' && (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Age
                    </label>
                    <input
                      type="number"
                      value={age}
                      onChange={(e) => setAge(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Alcohol Consuming
                    </label>
                    <select
                      value={alcoholConsuming}
                      onChange={(e) => setAlcoholConsuming(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Allergy
                    </label>
                    <select
                      value={allergy}
                      onChange={(e) => setAllergy(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Peer Pressure
                    </label>
                    <select
                      value={peerPressure}
                      onChange={(e) => setPeerPressure(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Yellow Fingers
                    </label>
                    <select
                      value={yellowFingers}
                      onChange={(e) => setYellowFingers(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                      Fatigue
                    </label>
                    <select
                      value={fatigue}
                      onChange={(e) => setFatigue(e.target.value)}
                      className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                      required
                    >
                      <option value="0">No</option>
                      <option value="1">Yes</option>
                    </select>
                  </div>
                </div>
                
                <div>
                  <label className="block text-sm font-medium text-[#2C3E50] dark:text-gray-300 mb-2">
                    Coughing
                  </label>
                  <select
                    value={coughing}
                    onChange={(e) => setCoughing(e.target.value)}
                    className="w-full px-4 py-2 rounded-lg border border-gray-300 focus:ring-2 focus:ring-[#4A90E2] focus:border-transparent dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    required
                  >
                    <option value="0">No</option>
                    <option value="1">Yes</option>
                  </select>
                </div>
              </div>
            )}

            <button
              type="submit"
              disabled={loading || !cancerType}
              className="w-full bg-[#4A90E2] hover:bg-[#357ABD] text-white font-semibold py-3 px-6 rounded-lg transition duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {loading ? 'Processing...' : 'Get Prediction'}
            </button>
          </form>
        </div>
      </div>
    </motion.div>
  );
};

export default Predict;