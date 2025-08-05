import React, { useState, useEffect } from 'react';

const CytotoxicityPredictionModel = () => {
  const [drugSmiles, setDrugSmiles] = useState('');
  const [selectedCellLines, setSelectedCellLines] = useState(['A549', 'MCF7', 'HCT116']);
  const [availableCellLines, setAvailableCellLines] = useState([]);
  const [cellLineCategories, setCellLineCategories] = useState({});
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [model2Info, setModel2Info] = useState(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  
  // Backend URL from environment
  const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';

  // Load Model 2 info and available cell lines on component mount
  useEffect(() => {
    const loadModel2Data = async () => {
      try {
        // Get Model 2 info
        const infoResponse = await fetch(`${backendUrl}/api/model2/info`);
        const info = await infoResponse.json();
        setModel2Info(info);

        // Get available cell lines
        const cellLinesResponse = await fetch(`${backendUrl}/api/model2/cell-lines`);
        const cellLinesData = await cellLinesResponse.json();
        setAvailableCellLines(cellLinesData.available_cell_lines || []);
        setCellLineCategories(cellLinesData.categories || {});
        
      } catch (error) {
        console.error('Error loading Model 2 data:', error);
      }
    };

    loadModel2Data();
  }, [backendUrl]);

  // SMILES examples for quick testing
  const smilesExamples = [
    {
      name: 'Aspirin',
      smiles: 'CC(=O)OC1=CC=CC=C1C(=O)O',
      description: 'Common pain reliever'
    },
    {
      name: 'Imatinib',
      smiles: 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C',
      description: 'BCR-ABL inhibitor for CML'
    },
    {
      name: 'Erlotinib',
      smiles: 'COc1cc2c(cc1OCCOC)ncnc2Nc3ccc(cc3)C#C',
      description: 'EGFR inhibitor'
    }
  ];

  const handlePrediction = async () => {
    if (!drugSmiles.trim()) {
      alert('Please enter a SMILES string');
      return;
    }

    setIsLoading(true);
    setPrediction(null);

    try {
      const requestBody = {
        smiles: drugSmiles.trim(),
        cell_lines: selectedCellLines
      };

      const endpoint = comparisonMode ? '/api/model2/compare' : '/api/model2/predict';
      const response = await fetch(`${backendUrl}${endpoint}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (response.status === 503) {
        // Training in progress
        const errorData = await response.json();
        setPrediction({
          training_in_progress: true,
          message: errorData.detail?.message || 'Model 2 training in progress',
          expected_availability: errorData.detail?.expected_availability || 'Soon'
        });
      } else if (response.ok) {
        const result = await response.json();
        setPrediction(result);
      } else {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Prediction failed');
      }

    } catch (error) {
      console.error('Prediction error:', error);
      setPrediction({
        error: true,
        message: error.message
      });
    } finally {
      setIsLoading(false);
    }
  };

  const toggleCellLineSelection = (cellLine) => {
    setSelectedCellLines(prev => {
      if (prev.includes(cellLine)) {
        return prev.filter(cl => cl !== cellLine);
      } else {
        return [...prev, cellLine];
      }
    });
  };

  const renderModelStatus = () => {
    if (!model2Info) return null;

    const statusColor = model2Info.model_loaded ? 'green' : model2Info.training_status === 'in_progress' ? 'yellow' : 'red';
    const statusText = model2Info.model_loaded ? 'Ready' : model2Info.training_status === 'in_progress' ? 'Training in Progress' : 'Not Available';

    return (
      <div className="mb-6 p-4 bg-gray-800 rounded-lg border border-gray-700">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-white">Model 2 Status</h3>
          <div className={`inline-flex items-center px-3 py-1 rounded-full text-sm font-medium ${
            statusColor === 'green' ? 'bg-green-900 text-green-300 border border-green-700' :
            statusColor === 'yellow' ? 'bg-yellow-900 text-yellow-300 border border-yellow-700' :
            'bg-red-900 text-red-300 border border-red-700'
          }`}>
            <div className={`w-2 h-2 rounded-full mr-2 ${
              statusColor === 'green' ? 'bg-green-400' :
              statusColor === 'yellow' ? 'bg-yellow-400 animate-pulse' :
              'bg-red-400'
            }`}></div>
            {statusText}
          </div>
        </div>
        
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm text-gray-300">
          <div>
            <span className="text-gray-400">Available Cell Lines:</span>
            <div className="font-medium">{model2Info.available_cell_lines}</div>
          </div>
          <div>
            <span className="text-gray-400">Data Sources:</span>
            <div className="font-medium">{model2Info.data_sources?.join(', ')}</div>
          </div>
          <div>
            <span className="text-gray-400">Prediction Type:</span>
            <div className="font-medium">{model2Info.prediction_type}</div>
          </div>
          <div>
            <span className="text-gray-400">Units:</span>
            <div className="font-medium">{model2Info.units}</div>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="p-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-purple-400 to-cyan-400 bg-clip-text text-transparent">
            🦠 Cancer Cell Line Cytotoxicity Predictor
          </h1>
          <p className="text-xl text-gray-300">
            Model 2: Predict drug cytotoxicity across cancer cell lines using comprehensive GDSC data
          </p>
        </div>

        {/* Model Status */}
        {renderModelStatus()}

        {/* Input Section */}
        <div className="bg-gray-800 rounded-lg border border-gray-700 mb-6">
          <div className="p-6 border-b border-gray-700">
            <h2 className="text-xl font-semibold text-white mb-4">Drug Input</h2>
            
            {/* SMILES Input */}
            <div className="mb-4">
              <label htmlFor="smiles" className="block text-sm font-medium text-gray-300 mb-2">
                SMILES String
              </label>
              <input
                id="smiles"
                type="text"
                value={drugSmiles}
                onChange={(e) => setDrugSmiles(e.target.value)}
                placeholder="Enter SMILES string (e.g., CC(=O)OC1=CC=CC=C1C(=O)O)"
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-md text-white placeholder-gray-400 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            {/* SMILES Examples */}
            <div className="mb-4">
              <div className="text-sm text-gray-400 mb-2">Quick Examples:</div>
              <div className="flex flex-wrap gap-2">
                {smilesExamples.map((example, index) => (
                  <button
                    key={index}
                    onClick={() => setDrugSmiles(example.smiles)}
                    className="px-3 py-1 bg-purple-600 hover:bg-purple-500 text-white text-sm rounded-md transition-colors"
                    title={example.description}
                  >
                    {example.name}
                  </button>
                ))}
              </div>
            </div>

            {/* Prediction Mode Toggle */}
            <div className="mb-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={comparisonMode}
                  onChange={(e) => setComparisonMode(e.target.checked)}
                  className="mr-2 rounded"
                />
                <span className="text-gray-300">Enable comparison analysis</span>
              </label>
            </div>
          </div>

          {/* Cell Line Selection */}
          <div className="p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Select Cancer Cell Lines</h3>
            
            {availableCellLines.length > 0 ? (
              <div>
                {/* All available cell lines in a grid */}
                <div className="mb-4">
                  <div className="text-sm font-medium text-gray-400 mb-2">
                    All Available Cell Lines ({availableCellLines.length})
                  </div>
                  <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
                    {availableCellLines.map(cellLine => (
                      <label key={cellLine} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={selectedCellLines.includes(cellLine)}
                          onChange={() => toggleCellLineSelection(cellLine)}
                          className="mr-2 rounded"
                        />
                        <span className="text-gray-300 text-sm">{cellLine}</span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-gray-400">Loading available cell lines...</div>
            )}

            <div className="mt-4 text-sm text-gray-400">
              Selected: {selectedCellLines.length} cell lines
            </div>
          </div>

          {/* Predict Button */}
          <div className="p-6 border-t border-gray-700">
            <button
              onClick={handlePrediction}
              disabled={isLoading || !drugSmiles.trim() || selectedCellLines.length === 0}
              className="w-full px-6 py-3 bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-700 hover:to-cyan-700 disabled:from-gray-600 disabled:to-gray-600 text-white font-medium rounded-lg transition-all disabled:cursor-not-allowed flex items-center justify-center"
            >
              {isLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Predicting...
                </>
              ) : (
                comparisonMode ? 'Compare Across Cell Lines' : 'Predict Cytotoxicity'
              )}
            </button>
          </div>
        </div>

        {/* Results Section */}
        {prediction && (
          <div className="bg-gray-800 rounded-lg border border-gray-700">
            {prediction.training_in_progress && (
              <div className="p-6 bg-yellow-900/20 border border-yellow-700 rounded-lg">
                <div className="flex items-center mb-4">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-yellow-400 mr-3"></div>
                  <h3 className="text-lg font-semibold text-yellow-300">Model 2 Training in Progress</h3>
                </div>
                <p className="text-yellow-200 mb-2">{prediction.message}</p>
                <p className="text-yellow-300 text-sm">Expected availability: {prediction.expected_availability}</p>
                <div className="mt-4 p-3 bg-yellow-800/30 rounded-lg">
                  <p className="text-yellow-100 text-sm">
                    🚀 <strong>What's happening:</strong> Model 2 is being trained on comprehensive cancer cell line data from GDSC1/GDSC2 
                    to achieve the target R² > 0.6 performance. Once training completes, you'll be able to predict drug cytotoxicity 
                    across {model2Info?.available_cell_lines || 36} different cancer cell lines.
                  </p>
                </div>
              </div>
            )}

            {prediction.error && (
              <div className="p-6 bg-red-900/20 border border-red-700 rounded-lg">
                <div className="flex items-center mb-4">
                  <span className="text-red-400 mr-3">⚠️</span>
                  <h3 className="text-lg font-semibold text-red-300">Prediction Error</h3>
                </div>
                <p className="text-red-200">{prediction.message}</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CytotoxicityPredictionModel;