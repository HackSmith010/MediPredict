import  { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { CheckCircle, AlertCircle, Download, ArrowLeft } from 'lucide-react';

interface PredictionResult {
  prediction: string;
  probability: number;
  status: string;
  type?: string; 
}

const Results = () => {
  const navigate = useNavigate();
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [isDownloading, setIsDownloading] = useState<boolean>(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  useEffect(() => {
    const storedResult = localStorage.getItem('predictionResult');
    const cancerType = localStorage.getItem('cancerType');
    
    if (storedResult) {
      const parsedResult = JSON.parse(storedResult);
      setResult({
        ...parsedResult,
        type: cancerType || 'breast' 
      });
    }
  }, []);

  const handleDownloadReport = async () => {
    if (!result) return;
    
    setIsDownloading(true);
    setDownloadError(null);
    
    try {
      const response = await fetch('http://localhost:5000/api/report/generate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(result),
      });
      
      if (!response.ok) {
        throw new Error('Failed to generate PDF report');
      }
      
      const blob = await response.blob();
      
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.style.display = 'none';
      a.href = url;
      a.download = `cancer_prediction_report_${new Date().toISOString().split('T')[0]}.pdf`;
      
      document.body.appendChild(a);
      a.click();
      
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
    } catch (error) {
      console.error('Error downloading report:', error);
      setDownloadError('Failed to generate PDF report. Please try again.');
    } finally {
      setIsDownloading(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="flex-1 py-12 bg-[#F5F7FA] dark:bg-gray-900"
    >
      <div className="max-w-2xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 text-center">
          <h1 className="text-3xl font-bold text-[#2C3E50] dark:text-white mb-6">
            Prediction Results
          </h1>

          {result ? (
            <div className="space-y-6">
              <div className="flex justify-center items-center space-x-3">
                {result.prediction.toLowerCase() === 'benign' ||
                result.prediction.toLowerCase() === 'no cancer' ? (
                  <CheckCircle className="text-green-500" size={48} />
                ) : (
                  <AlertCircle className="text-red-500" size={48} />
                )}
                <h2
                  className={`text-2xl font-semibold ${
                    result.prediction.toLowerCase() === 'benign' ||
                    result.prediction.toLowerCase() === 'no cancer'
                      ? 'text-green-600'
                      : 'text-red-600'
                  }`}
                >
                  {result.prediction}
                </h2>
              </div>

              <p className="text-lg text-gray-700 dark:text-gray-300">
                Probability: <strong>{(result.probability * 100).toFixed(2)}%</strong>
              </p>

              <div className="bg-gray-100 dark:bg-gray-700 p-4 rounded-lg">
                <h3 className="text-lg font-semibold text-[#2C3E50] dark:text-white mb-2">
                  Recommendations
                </h3>
                {result.prediction.toLowerCase() === 'benign' ||
                result.prediction.toLowerCase() === 'no cancer' ? (
                  <ul className="text-gray-700 dark:text-gray-300 text-left list-disc pl-5">
                    <li>Maintain a healthy lifestyle and regular check-ups.</li>
                    <li>Stay active and eat a balanced diet.</li>
                    <li>Continue routine screenings for early detection.</li>
                  </ul>
                ) : (
                  <ul className="text-gray-700 dark:text-gray-300 text-left list-disc pl-5">
                    <li>Consult with a healthcare professional immediately.</li>
                    <li>Follow the prescribed treatment plan.</li>
                    <li>Consider additional diagnostic tests for confirmation.</li>
                    <li>Seek support from family, friends, or support groups.</li>
                  </ul>
                )}
              </div>

              {downloadError && (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
                  {downloadError}
                </div>
              )}

              <div className="flex flex-col sm:flex-row justify-center gap-4">
                <button
                  onClick={handleDownloadReport}
                  disabled={isDownloading}
                  className={`flex items-center justify-center gap-2 bg-[#4A90E2] hover:bg-[#357ABD] text-white font-semibold py-3 px-6 rounded-lg transition duration-300 ${
                    isDownloading ? 'opacity-50 cursor-not-allowed' : ''
                  }`}
                >
                  {isDownloading ? (
                    <>
                      <svg className="animate-spin -ml-1 mr-2 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                      </svg>
                      Generating PDF...
                    </>
                  ) : (
                    <>
                      <Download size={18} />
                      Download Report (PDF)
                    </>
                  )}
                </button>

                <button
                  onClick={() => navigate('/predict')}
                  className="flex items-center justify-center gap-2 bg-gray-600 hover:bg-gray-500 text-white font-semibold py-3 px-6 rounded-lg transition duration-300"
                >
                  <ArrowLeft size={18} />
                  Make Another Prediction
                </button>
              </div>
            </div>
          ) : (
            <p className="text-lg text-gray-700 dark:text-gray-300">
              No prediction results found. Please make a prediction first.
            </p>
          )}
        </div>
      </div>
    </motion.div>
  );
};

export default Results;