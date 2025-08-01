import React, { useState, useEffect } from 'react';

const CellLineResponseModel = () => {
  const [drugSmiles, setDrugSmiles] = useState('');
  const [drugName, setDrugName] = useState('');
  const [selectedCellLine, setSelectedCellLine] = useState('A549');
  const [cancerType, setCancerType] = useState('LUNG');
  const [genomicFeatures, setGenomicFeatures] = useState({
    mutations: {
      TP53: 0,
      KRAS: 0,
      EGFR: 0,
      BRAF: 0,
      PIK3CA: 0,
      PTEN: 0,
      BRCA1: 0,
      BRCA2: 0
    },
    cnvs: {
      MYC: 0,
      CDKN2A: 0,
      PTEN: 0,
      EGFR: 0,
      HER2: 0,
      MDM2: 0
    },
    expression: {
      EGFR: 0,
      KRAS: 0,
      TP53: 0,
      MYC: 0,
      PTEN: 0,
      HER2: 0
    }
  });
  
  const [prediction, setPrediction] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [examples, setExamples] = useState(null);
  const [comparisonMode, setComparisonMode] = useState(false);
  const [comparisonCellLines, setComparisonCellLines] = useState([]);
  const [comparisonResults, setComparisonResults] = useState(null);

  // Predefined cell line profiles
  const cellLineProfiles = {
    A549: {
      name: 'A549',
      cancer_type: 'LUNG',
      description: 'Lung adenocarcinoma cell line',
      genomic_features: {
        mutations: { TP53: 1, KRAS: 1, EGFR: 0, BRAF: 0, PIK3CA: 0, PTEN: 0, BRCA1: 0, BRCA2: 0 },
        cnvs: { MYC: 1, CDKN2A: -1, PTEN: 0, EGFR: 0, HER2: 0, MDM2: 0 },
        expression: { EGFR: -0.5, KRAS: 1.2, TP53: -1.8, MYC: 0.8, PTEN: -0.3, HER2: -0.2 }
      }
    },
    MCF7: {
      name: 'MCF7',
      cancer_type: 'BREAST',
      description: 'Breast adenocarcinoma cell line',
      genomic_features: {
        mutations: { TP53: 0, KRAS: 0, EGFR: 0, BRAF: 0, PIK3CA: 1, PTEN: 0, BRCA1: 0, BRCA2: 0 },
        cnvs: { MYC: 0, CDKN2A: 0, PTEN: 0, EGFR: 0, HER2: 0, MDM2: 0 },
        expression: { EGFR: 0.3, KRAS: -0.2, TP53: 0.8, MYC: 0.1, PTEN: 0.5, HER2: 0.4 }
      }
    },
    HCT116: {
      name: 'HCT116',
      cancer_type: 'COLON',
      description: 'Colorectal carcinoma cell line',
      genomic_features: {
        mutations: { TP53: 0, KRAS: 1, EGFR: 0, BRAF: 0, PIK3CA: 1, PTEN: 0, BRCA1: 0, BRCA2: 0 },
        cnvs: { MYC: 1, CDKN2A: 0, PTEN: -1, EGFR: 0, HER2: 0, MDM2: 0 },
        expression: { EGFR: 1.5, KRAS: 2.0, TP53: 0.5, MYC: 1.2, PTEN: -1.5, HER2: 0.1 }
      }
    }
  };

  // Example drugs
  const exampleDrugs = [
    {
      name: 'Erlotinib',
      smiles: 'Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC',
      target: 'EGFR'
    },
    {
      name: 'Trametinib',
      smiles: 'CC(C)[C@H](C(=O)N1CC2=C(C1=O)SC(=N2)C3=CC(=C(C=C3)F)F)N4C(=O)C5=C(C4=O)C=CC=C5I',
      target: 'MEK1/2'
    },
    {
      name: 'Imatinib',
      smiles: 'Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)CN5CCN(CC5)C',
      target: 'BCR-ABL'
    }
  ];

  useEffect(() => {
    // Load examples on component mount
    loadExamples();
  }, []);

  const loadExamples = async () => {
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/cell-line/examples`);
      if (response.ok) {
        const data = await response.json();
        setExamples(data);
      }
    } catch (error) {
      console.error('Error loading examples:', error);
    }
  };

  const loadCellLineProfile = (cellLineName) => {
    const profile = cellLineProfiles[cellLineName];
    if (profile) {
      setSelectedCellLine(cellLineName);
      setCancerType(profile.cancer_type);
      setGenomicFeatures(profile.genomic_features);
    }
  };

  const loadExampleDrug = (drug) => {
    setDrugName(drug.name);
    setDrugSmiles(drug.smiles);
  };

  const predictDrugSensitivity = async () => {
    if (!drugSmiles.trim()) {
      alert('Please enter a SMILES string for the drug');
      return;
    }

    setIsLoading(true);
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      const response = await fetch(`${backendUrl}/api/cell-line/predict`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          smiles: drugSmiles,
          drug_name: drugName || 'Unknown Drug',
          cell_line: {
            cell_line_name: selectedCellLine,
            cancer_type: cancerType,
            genomic_features: genomicFeatures
          }
        })
      });

      if (response.ok) {
        const result = await response.json();
        setPrediction(result);
      } else {
        const error = await response.json();
        alert(`Prediction failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  const compareCellLines = async () => {
    if (!drugSmiles.trim()) {
      alert('Please enter a SMILES string for the drug');
      return;
    }

    if (comparisonCellLines.length < 2) {
      alert('Please select at least 2 cell lines for comparison');
      return;
    }

    setIsLoading(true);
    try {
      const backendUrl = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8001';
      
      const cellLinesData = comparisonCellLines.map(cellLineName => {
        const profile = cellLineProfiles[cellLineName];
        return {
          cell_line_name: cellLineName,
          cancer_type: profile.cancer_type,
          genomic_features: profile.genomic_features
        };
      });

      const response = await fetch(`${backendUrl}/api/cell-line/compare`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          smiles: drugSmiles,
          drug_name: drugName || 'Unknown Drug',
          cell_lines: cellLinesData
        })
      });

      if (response.ok) {
        const result = await response.json();
        setComparisonResults(result);
      } else {
        const error = await response.json();
        alert(`Comparison failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Comparison error:', error);
      alert('Comparison failed. Please check your connection.');
    } finally {
      setIsLoading(false);
    }
  };

  const getSensitivityColor = (sensitivityClass) => {
    switch (sensitivityClass) {
      case 'SENSITIVE': return 'text-green-600 bg-green-50';
      case 'MODERATE': return 'text-yellow-600 bg-yellow-50';
      case 'RESISTANT': return 'text-red-600 bg-red-50';
      default: return 'text-gray-600 bg-gray-50';
    }
  };

  const formatIC50 = (ic50_nm) => {
    if (ic50_nm < 1000) {
      return `${ic50_nm.toFixed(1)} nM`;
    } else if (ic50_nm < 1000000) {
      return `${(ic50_nm / 1000).toFixed(1)} μM`;
    } else {
      return `${(ic50_nm / 1000000).toFixed(1)} mM`;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-800 to-blue-800 p-6">
        <h1 className="text-4xl font-bold mb-2">🧬 Cell Line Response Model</h1>
        <p className="text-lg text-purple-200">
          Multi-modal IC₅₀ prediction combining drug molecular structure and cancer cell line genomics
        </p>
      </div>

      <div className="container mx-auto p-6">
        {/* Mode Selection */}
        <div className="mb-6">
          <div className="flex space-x-4">
            <button
              onClick={() => setComparisonMode(false)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                !comparisonMode
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              🎯 Single Prediction
            </button>
            <button
              onClick={() => setComparisonMode(true)}
              className={`px-6 py-3 rounded-lg font-semibold transition-all ${
                comparisonMode
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              📊 Cell Line Comparison
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Input Panel */}
          <div className="bg-gray-800 rounded-xl p-6 shadow-xl">
            <h2 className="text-2xl font-bold mb-4 text-blue-400">
              {comparisonMode ? '📊 Comparison Setup' : '🎯 Prediction Setup'}
            </h2>

            {/* Drug Input */}
            <div className="mb-6">
              <h3 className="text-lg font-semibold mb-3 text-green-400">💊 Drug Information</h3>
              
              {/* Example Drugs */}
              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Quick Select Drug:</label>
                <div className="grid grid-cols-1 gap-2">
                  {exampleDrugs.map((drug, index) => (
                    <button
                      key={index}
                      onClick={() => loadExampleDrug(drug)}
                      className="bg-gray-700 hover:bg-gray-600 p-3 rounded-lg text-left transition-all"
                    >
                      <div className="font-semibold">{drug.name}</div>
                      <div className="text-sm text-gray-400">Target: {drug.target}</div>
                    </button>
                  ))}
                </div>
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">Drug Name (Optional):</label>
                <input
                  type="text"
                  value={drugName}
                  onChange={(e) => setDrugName(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Erlotinib"
                />
              </div>

              <div className="mb-4">
                <label className="block text-sm font-medium mb-2">SMILES String:</label>
                <textarea
                  value={drugSmiles}
                  onChange={(e) => setDrugSmiles(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 h-20 focus:ring-2 focus:ring-blue-500"
                  placeholder="e.g., Cc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCOCCOC"
                />
              </div>
            </div>

            {!comparisonMode ? (
              /* Single Prediction Mode */
              <>
                {/* Cell Line Selection */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-3 text-purple-400">🧬 Cell Line Profile</h3>
                  
                  <div className="mb-4">
                    <label className="block text-sm font-medium mb-2">Select Cell Line:</label>
                    <div className="grid grid-cols-1 gap-2">
                      {Object.keys(cellLineProfiles).map((cellLineName) => {
                        const profile = cellLineProfiles[cellLineName];
                        return (
                          <button
                            key={cellLineName}
                            onClick={() => loadCellLineProfile(cellLineName)}
                            className={`p-3 rounded-lg text-left transition-all ${
                              selectedCellLine === cellLineName
                                ? 'bg-purple-600 border-2 border-purple-400'
                                : 'bg-gray-700 hover:bg-gray-600 border-2 border-transparent'
                            }`}
                          >
                            <div className="font-semibold">{profile.name}</div>
                            <div className="text-sm text-gray-400">{profile.description}</div>
                            <div className="text-xs text-gray-500 mt-1">
                              {profile.cancer_type} | Key mutations: {
                                Object.entries(profile.genomic_features.mutations)
                                  .filter(([gene, status]) => status === 1)
                                  .map(([gene]) => gene)
                                  .join(', ') || 'None'
                              }
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  </div>
                </div>

                {/* Genomic Features Editor */}
                <div className="mb-6">
                  <h3 className="text-lg font-semibold mb-3 text-orange-400">🧬 Genomic Features</h3>
                  
                  {/* Mutations */}
                  <div className="mb-4">
                    <h4 className="font-medium mb-2">Mutations (0 = Wild-type, 1 = Mutated):</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(genomicFeatures.mutations).map(([gene, value]) => (
                        <div key={gene} className="flex items-center justify-between bg-gray-700 p-2 rounded">
                          <span className="text-sm">{gene}:</span>
                          <select
                            value={value}
                            onChange={(e) => setGenomicFeatures(prev => ({
                              ...prev,
                              mutations: { ...prev.mutations, [gene]: parseInt(e.target.value) }
                            }))}
                            className="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm"
                          >
                            <option value={0}>WT</option>
                            <option value={1}>MUT</option>
                          </select>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* CNVs */}
                  <div className="mb-4">
                    <h4 className="font-medium mb-2">Copy Number Variations (-1 = Deletion, 0 = Normal, 1 = Amplification):</h4>
                    <div className="grid grid-cols-2 gap-2">
                      {Object.entries(genomicFeatures.cnvs).map(([gene, value]) => (
                        <div key={gene} className="flex items-center justify-between bg-gray-700 p-2 rounded">
                          <span className="text-sm">{gene}:</span>
                          <select
                            value={value}
                            onChange={(e) => setGenomicFeatures(prev => ({
                              ...prev,
                              cnvs: { ...prev.cnvs, [gene]: parseInt(e.target.value) }
                            }))}
                            className="bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm"
                          >
                            <option value={-1}>DEL</option>
                            <option value={0}>NORM</option>
                            <option value={1}>AMP</option>
                          </select>
                        </div>
                      ))}
                    </div>
                  </div>

                  {/* Expression */}
                  <div className="mb-4">
                    <h4 className="font-medium mb-2">Expression Levels (Z-scores):</h4>
                    <div className="grid grid-cols-1 gap-2">
                      {Object.entries(genomicFeatures.expression).map(([gene, value]) => (
                        <div key={gene} className="flex items-center justify-between bg-gray-700 p-2 rounded">
                          <span className="text-sm">{gene}:</span>
                          <input
                            type="number"
                            value={value}
                            onChange={(e) => setGenomicFeatures(prev => ({
                              ...prev,
                              expression: { ...prev.expression, [gene]: parseFloat(e.target.value) || 0 }
                            }))}
                            step="0.1"
                            min="-5"
                            max="5"
                            className="bg-gray-600 border border-gray-500 rounded px-3 py-1 text-sm w-20"
                          />
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </>
            ) : (
              /* Comparison Mode */
              <div className="mb-6">
                <h3 className="text-lg font-semibold mb-3 text-purple-400">🧬 Select Cell Lines for Comparison</h3>
                
                <div className="space-y-2">
                  {Object.keys(cellLineProfiles).map((cellLineName) => {
                    const profile = cellLineProfiles[cellLineName];
                    const isSelected = comparisonCellLines.includes(cellLineName);
                    
                    return (
                      <label key={cellLineName} className="flex items-center">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={(e) => {
                            if (e.target.checked) {
                              setComparisonCellLines(prev => [...prev, cellLineName]);
                            } else {
                              setComparisonCellLines(prev => prev.filter(cl => cl !== cellLineName));
                            }
                          }}
                          className="mr-3 text-blue-600"
                        />
                        <div className="flex-1 bg-gray-700 p-3 rounded-lg">
                          <div className="font-semibold">{profile.name}</div>
                          <div className="text-sm text-gray-400">{profile.description}</div>
                          <div className="text-xs text-gray-500 mt-1">
                            Key mutations: {
                              Object.entries(profile.genomic_features.mutations)
                                .filter(([gene, status]) => status === 1)
                                .map(([gene]) => gene)
                                .join(', ') || 'None'
                            }
                          </div>
                        </div>
                      </label>
                    );
                  })}
                </div>
                
                <div className="mt-3 text-sm text-gray-400">
                  Selected: {comparisonCellLines.length} cell lines
                </div>
              </div>
            )}

            {/* Predict Button */}
            <button
              onClick={comparisonMode ? compareCellLines : predictDrugSensitivity}
              disabled={isLoading}
              className="w-full bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-bold py-4 px-6 rounded-lg transition-all shadow-lg"
            >
              {isLoading ? (
                <span className="flex items-center justify-center">
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  {comparisonMode ? 'Comparing Cell Lines...' : 'Predicting...'}
                </span>
              ) : (
                comparisonMode ? '📊 Compare Cell Lines' : '🎯 Predict Drug Sensitivity'
              )}
            </button>
          </div>

          {/* Results Panel */}
          <div className="bg-gray-800 rounded-xl p-6 shadow-xl">
            <h2 className="text-2xl font-bold mb-4 text-blue-400">
              {comparisonMode ? '📊 Comparison Results' : '🎯 Prediction Results'}
            </h2>

            {!comparisonMode && prediction && (
              /* Single Prediction Results */
              <div className="space-y-4">
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-2">Drug Sensitivity Prediction</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-400">Drug</div>
                      <div className="font-semibold">{prediction.drug_name}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Cell Line</div>
                      <div className="font-semibold">{prediction.cell_line_name} ({prediction.cancer_type})</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Predicted IC₅₀</div>
                      <div className="font-bold text-xl">{formatIC50(prediction.predicted_ic50_nm)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">pIC₅₀</div>
                      <div className="font-bold text-xl">{prediction.predicted_pic50.toFixed(2)}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Sensitivity</div>
                      <div className={`inline-block px-3 py-1 rounded-full text-sm font-semibold ${getSensitivityColor(prediction.sensitivity_class)}`}>
                        {prediction.sensitivity_class}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Confidence</div>
                      <div className="font-semibold">{(prediction.confidence * 100).toFixed(1)}%</div>
                    </div>
                  </div>
                </div>

                {/* Genomic Context */}
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-2 text-orange-400">🧬 Genomic Context</h3>
                  <div className="space-y-2">
                    {prediction.genomic_context.key_mutations.length > 0 && (
                      <div>
                        <span className="text-red-400 font-medium">Key Mutations:</span>
                        <span className="ml-2">{prediction.genomic_context.key_mutations.join(', ')}</span>
                      </div>
                    )}
                    {prediction.genomic_context.amplifications.length > 0 && (
                      <div>
                        <span className="text-green-400 font-medium">Amplifications:</span>
                        <span className="ml-2">{prediction.genomic_context.amplifications.join(', ')}</span>
                      </div>
                    )}
                    {prediction.genomic_context.deletions.length > 0 && (
                      <div>
                        <span className="text-red-400 font-medium">Deletions:</span>
                        <span className="ml-2">{prediction.genomic_context.deletions.join(', ')}</span>
                      </div>
                    )}
                    {prediction.genomic_context.high_expression.length > 0 && (
                      <div>
                        <span className="text-blue-400 font-medium">High Expression:</span>
                        <span className="ml-2">{prediction.genomic_context.high_expression.join(', ')}</span>
                      </div>
                    )}
                    {prediction.genomic_context.low_expression.length > 0 && (
                      <div>
                        <span className="text-yellow-400 font-medium">Low Expression:</span>
                        <span className="ml-2">{prediction.genomic_context.low_expression.join(', ')}</span>
                      </div>
                    )}
                  </div>
                </div>

                {/* Clinical Insights */}
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-2 text-green-400">💡 Clinical Insights</h3>
                  <div className="text-sm text-gray-300">
                    {prediction.sensitivity_class === 'RESISTANT' && prediction.genomic_context.key_mutations.includes('KRAS') && prediction.drug_name?.toLowerCase().includes('erlotinib') && (
                      <p className="mb-2">⚠️ <strong>KRAS mutation confers resistance to EGFR inhibitors like Erlotinib.</strong></p>
                    )}
                    {prediction.sensitivity_class === 'SENSITIVE' && prediction.genomic_context.key_mutations.includes('KRAS') && prediction.drug_name?.toLowerCase().includes('trametinib') && (
                      <p className="mb-2">✅ <strong>KRAS mutation drives MEK pathway dependence, making Trametinib effective.</strong></p>
                    )}
                    {prediction.genomic_context.key_mutations.includes('TP53') && (
                      <p className="mb-2">ℹ️ <strong>p53 mutation may reduce apoptotic response to therapy.</strong></p>
                    )}
                    <p>
                      Prediction confidence: {(prediction.confidence * 100).toFixed(1)}% 
                      (Based on genomic complexity: {prediction.genomic_context.key_mutations.length} key mutations)
                    </p>
                  </div>
                </div>
              </div>
            )}

            {comparisonMode && comparisonResults && (
              /* Comparison Results */
              <div className="space-y-4">
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-2">Comparison Summary</h3>
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-gray-400">Drug</div>
                      <div className="font-semibold">{comparisonResults.drug_name}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Cell Lines</div>
                      <div className="font-semibold">{comparisonResults.summary.total_cell_lines}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">IC₅₀ Range</div>
                      <div className="font-semibold">
                        {formatIC50(comparisonResults.summary.ic50_range.min_nm)} - {formatIC50(comparisonResults.summary.ic50_range.max_nm)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Fold Difference</div>
                      <div className="font-semibold">{comparisonResults.summary.ic50_range.fold_difference.toFixed(1)}x</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Most Sensitive</div>
                      <div className="font-semibold text-green-400">{comparisonResults.summary.most_sensitive}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-400">Most Resistant</div>
                      <div className="font-semibold text-red-400">{comparisonResults.summary.most_resistant}</div>
                    </div>
                  </div>
                </div>

                {/* Individual Results */}
                <div className="space-y-3">
                  <h3 className="text-lg font-semibold text-purple-400">Individual Cell Line Results</h3>
                  {comparisonResults.predictions
                    .sort((a, b) => a.predicted_ic50_nm - b.predicted_ic50_nm)
                    .map((pred, index) => (
                    <div key={index} className="bg-gray-700 p-4 rounded-lg">
                      <div className="flex justify-between items-center mb-2">
                        <h4 className="font-semibold">{pred.cell_line_name} ({pred.cancer_type})</h4>
                        <div className={`px-3 py-1 rounded-full text-sm font-semibold ${getSensitivityColor(pred.sensitivity_class)}`}>
                          {pred.sensitivity_class}
                        </div>
                      </div>
                      
                      <div className="grid grid-cols-3 gap-4 text-sm">
                        <div>
                          <div className="text-gray-400">IC₅₀</div>
                          <div className="font-semibold">{formatIC50(pred.predicted_ic50_nm)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">pIC₅₀</div>
                          <div className="font-semibold">{pred.predicted_pic50.toFixed(2)}</div>
                        </div>
                        <div>
                          <div className="text-gray-400">Confidence</div>
                          <div className="font-semibold">{(pred.confidence * 100).toFixed(1)}%</div>
                        </div>
                      </div>
                      
                      {pred.genomic_context.key_mutations.length > 0 && (
                        <div className="mt-2 text-xs">
                          <span className="text-red-400">Key mutations:</span>
                          <span className="ml-1">{pred.genomic_context.key_mutations.join(', ')}</span>
                        </div>
                      )}
                    </div>
                  ))}
                </div>

                {/* Sensitivity Distribution */}
                <div className="bg-gray-700 p-4 rounded-lg">
                  <h3 className="text-lg font-semibold mb-2 text-green-400">Sensitivity Distribution</h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold text-green-400">{comparisonResults.summary.sensitivity_distribution.sensitive}</div>
                      <div className="text-sm text-gray-400">Sensitive</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-yellow-400">{comparisonResults.summary.sensitivity_distribution.moderate}</div>
                      <div className="text-sm text-gray-400">Moderate</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold text-red-400">{comparisonResults.summary.sensitivity_distribution.resistant}</div>
                      <div className="text-sm text-gray-400">Resistant</div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {!prediction && !comparisonResults && (
              <div className="text-center text-gray-400 py-8">
                <div className="text-6xl mb-4">🧬</div>
                <p>Enter drug and genomic information above to get AI-powered predictions</p>
                <p className="text-sm mt-2">Multi-modal model combines molecular structure and genomics for precision oncology</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default CellLineResponseModel;