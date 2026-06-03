import { useState } from "react";
import axios from "axios";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [activeTab, setActiveTab] = useState("train");

  const [modelName, setModelName] = useState("resnet18");
  const [epochs, setEpochs] = useState(2);
  const [trainSplit, setTrainSplit] = useState(0.8);
  const [lr, setLr] = useState(0.001);

  const [modelId, setModelId] = useState("");
  const [trainingResult, setTrainingResult] = useState(null);
  const [showHistory, setShowHistory] = useState(false);

  const [file, setFile] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);

  const [successMessage, setSuccessMessage] = useState("");
  const [errorMessage, setErrorMessage] = useState("");

  const clearMessages = () => {
    setSuccessMessage("");
    setErrorMessage("");
  };

  const getErrorMessage = (error) => {
    if (error.response && error.response.data) {
      if (typeof error.response.data.detail === "string") {
        return error.response.data.detail;
      }

      if (Array.isArray(error.response.data.detail)) {
        return error.response.data.detail.map((item) => item.msg).join(", ");
      }

      return JSON.stringify(error.response.data);
    }

    if (error.message) {
      return error.message;
    }

    return "Something went wrong";
  };

  const trainModel = async () => {
    clearMessages();
    setTrainingResult(null);
    setPrediction(null);

    if (!modelName || !epochs || !trainSplit || !lr) {
      setErrorMessage("Please fill all training fields.");
      return;
    }

    if (trainSplit <= 0 || trainSplit >= 1) {
      setErrorMessage("Train split must be between 0 and 1, example: 0.8");
      return;
    }

    try {
      setIsTraining(true);
      setSuccessMessage("Training started. Please wait...");

      const formData = new FormData();
      formData.append("model_name", modelName);
      formData.append("epochs", epochs);
      formData.append("train_split", trainSplit);
      formData.append("lr", lr);

      const response = await axios.post(`${API_URL}/train`, formData);

      setTrainingResult(response.data);
      setModelId(response.data.model_id);
      setSuccessMessage("Training completed successfully.");
      setActiveTab("results");
    } catch (error) {
      setErrorMessage(getErrorMessage(error));
    } finally {
      setIsTraining(false);
    }
  };

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];

    setFile(selectedFile);
    setPrediction(null);
    clearMessages();

    if (selectedFile) {
      setImagePreview(URL.createObjectURL(selectedFile));
    } else {
      setImagePreview(null);
    }
  };

  const predictImage = async () => {
    clearMessages();
    setPrediction(null);

    if (!modelId) {
      setErrorMessage("Please train a model first or enter a valid Model ID.");
      return;
    }

    if (!file) {
      setErrorMessage("Please upload an image first.");
      return;
    }

    try {
      setIsPredicting(true);
      setSuccessMessage("Prediction started. Please wait...");

      const formData = new FormData();
      formData.append("model_id", modelId);
      formData.append("file", file);

      const response = await axios.post(`${API_URL}/predict`, formData);

      setPrediction(response.data);
      setSuccessMessage("Prediction completed successfully.");
    } catch (error) {
      setErrorMessage(getErrorMessage(error));
    } finally {
      setIsPredicting(false);
    }
  };

  const latestHistory =
    trainingResult && trainingResult.history.length > 0
      ? trainingResult.history[trainingResult.history.length - 1]
      : null;

  return (
    <div style={styles.page}>
      <div style={styles.container}>
        <div style={styles.header}>
          <div>
            <h1 style={styles.title}>Skin Cancer Detection</h1>
            <p style={styles.subtitle}>
              Multi-class image classification using ResNet and VGG models
            </p>
          </div>

          <div style={styles.statusPill}>
            Backend: {isTraining || isPredicting ? "Processing" : "Ready"}
          </div>
        </div>

        {successMessage && <div style={styles.successBox}>{successMessage}</div>}
        {errorMessage && <div style={styles.errorBox}>{errorMessage}</div>}

        <div style={styles.tabs}>
          <button
            style={activeTab === "train" ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab("train")}
          >
            Train Model
          </button>

          <button
            style={activeTab === "predict" ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab("predict")}
          >
            Predict Image
          </button>

          <button
            style={activeTab === "results" ? styles.activeTab : styles.tab}
            onClick={() => setActiveTab("results")}
          >
            Training Results
          </button>
        </div>

        {activeTab === "train" && (
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <div>
                <h2>Train Model</h2>
                <p style={styles.mutedText}>
                  Select model, epochs and learning rate to start training.
                </p>
              </div>
            </div>

            <div style={styles.grid}>
              <div>
                <label style={styles.label}>Model</label>
                <select
                  style={styles.input}
                  value={modelName}
                  disabled={isTraining}
                  onChange={(e) => setModelName(e.target.value)}
                >
                  <option value="resnet18">ResNet18</option>
                  <option value="resnet50">ResNet50</option>
                  <option value="vgg16">VGG16</option>
                  <option value="densenet121">DenseNet121</option>
<option value="mobilenetv2">MobileNetV2</option>
<option value="efficientnetb0">EfficientNetB0</option>
<option value="swin">Swin Transformer</option>
<option value="convnext">ConvNeXt Tiny</option>
<option value="vit">Vision Transformer (ViT)</option>
<option value="alexnet">AlexNet</option>

                </select>
              </div>

              <div>
                <label style={styles.label}>Epochs</label>
                <input
                  style={styles.input}
                  type="number"
                  value={epochs}
                  disabled={isTraining}
                  onChange={(e) => setEpochs(e.target.value)}
                />
              </div>

              <div>
                <label style={styles.label}>Train Split</label>
                <input
                  style={styles.input}
                  type="number"
                  step="0.1"
                  value={trainSplit}
                  disabled={isTraining}
                  onChange={(e) => setTrainSplit(e.target.value)}
                />
              </div>

              <div>
                <label style={styles.label}>Learning Rate</label>
                <input
                  style={styles.input}
                  type="number"
                  step="0.0001"
                  value={lr}
                  disabled={isTraining}
                  onChange={(e) => setLr(e.target.value)}
                />
              </div>
            </div>

            <button
              style={isTraining ? styles.disabledButton : styles.button}
              onClick={trainModel}
              disabled={isTraining}
            >
              {isTraining ? "Training in progress..." : "Start Training"}
            </button>

            {isTraining && (
              <div style={styles.loaderBox}>
                <div style={styles.spinner}></div>
                <div>
                  <b>Training is running...</b>
                  <p style={styles.mutedText}>
                    Check backend terminal for live epoch logs.
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === "predict" && (
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <div>
                <h2>Predict Image</h2>
                <p style={styles.mutedText}>
                  Upload a skin lesion image and predict its class.
                </p>
              </div>
            </div>

            <label style={styles.label}>Model ID</label>
            <input
              style={styles.input}
              type="text"
              placeholder="Paste trained model id here"
              value={modelId}
              disabled={isPredicting}
              onChange={(e) => setModelId(e.target.value)}
            />

            <label style={styles.label}>Upload Image</label>
            <input
              style={styles.input}
              type="file"
              accept="image/*"
              disabled={isPredicting}
              onChange={handleFileChange}
            />

            {imagePreview && (
              <div style={styles.previewWrapper}>
                <img src={imagePreview} alt="Preview" style={styles.previewImage} />

                <div>
                  <h3>Selected Image</h3>
                  <p style={styles.mutedText}>{file?.name}</p>
                </div>
              </div>
            )}

            <button
              style={isPredicting ? styles.disabledButton : styles.button}
              onClick={predictImage}
              disabled={isPredicting}
            >
              {isPredicting ? "Prediction in progress..." : "Predict Image"}
            </button>

            {isPredicting && (
              <div style={styles.loaderBox}>
                <div style={styles.spinner}></div>
                <p>Prediction is running...</p>
              </div>
            )}

            {prediction && (
              <div style={styles.predictionCard}>
                <p style={styles.mutedText}>Prediction Result</p>
                <h2>{prediction.prediction}</h2>
                <p>
                  Confidence:{" "}
                  <b>{(prediction.confidence * 100).toFixed(2)}%</b>
                </p>
              </div>
            )}
          </div>
        )}

        {activeTab === "results" && (
          <div style={styles.card}>
            <div style={styles.cardHeader}>
              <div>
                <h2>Training Results</h2>
                <p style={styles.mutedText}>
                  View latest trained model details and epoch history.
                </p>
              </div>
            </div>

            {!trainingResult && (
              <div style={styles.emptyBox}>
                No training result available yet. Please train a model first.
              </div>
            )}

            {trainingResult && (
              <>
                <div style={styles.summaryGrid}>
                  <div style={styles.metricCard}>
                    <p>Model ID</p>
                    <b style={styles.smallText}>{trainingResult.model_id}</b>
                  </div>

                  <div style={styles.metricCard}>
                    <p>Final Train Accuracy</p>
                    <b>{latestHistory?.train_accuracy?.toFixed(2)}%</b>
                  </div>

                  <div style={styles.metricCard}>
                    <p>Final Validation Accuracy</p>
                    <b>{latestHistory?.val_accuracy?.toFixed(2)}%</b>
                  </div>

                  <div style={styles.metricCard}>
                    <p>Final Loss</p>
                    <b>{latestHistory?.loss?.toFixed(4)}</b>
                  </div>
                </div>

                <div style={styles.pathBox}>
                  <b>Model Path:</b> {trainingResult.model_path}
                </div>

                <button
                  style={styles.secondaryButton}
                  onClick={() => setShowHistory(!showHistory)}
                >
                  {showHistory ? "Hide History" : "Show History"}
                </button>

                {showHistory && (
                  <div style={styles.tableWrapper}>
                    <table style={styles.table}>
                      <thead>
                        <tr>
                          <th style={styles.th}>Epoch</th>
                          <th style={styles.th}>Loss</th>
                          <th style={styles.th}>Train Accuracy</th>
                          <th style={styles.th}>Validation Loss</th>
                          <th style={styles.th}>Validation Accuracy</th>
                        </tr>
                      </thead>

                      <tbody>
                        {trainingResult.history.map((item) => (
                          <tr key={item.epoch}>
                            <td style={styles.td}>{item.epoch}</td>
                            <td style={styles.td}>{item.loss?.toFixed(4)}</td>
                            <td style={styles.td}>
                              {item.train_accuracy?.toFixed(2)}%
                            </td>
                            <td style={styles.td}>{item.val_loss?.toFixed(4)}</td>
                            <td style={styles.td}>
                              {item.val_accuracy?.toFixed(2)}%
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

const spinAnimation = `
@keyframes spin {
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
}
`;

const styleSheet = document.createElement("style");
styleSheet.innerText = spinAnimation;
document.head.appendChild(styleSheet);

const styles = {
  page: {
    minHeight: "100vh",
    background: "linear-gradient(135deg, #eef2ff, #f8fafc)",
    padding: "30px",
    fontFamily: "Arial, sans-serif",
    color: "#111827",
  },
  container: {
    maxWidth: "1000px",
    margin: "0 auto",
  },
  header: {
    background: "white",
    padding: "25px",
    borderRadius: "18px",
    boxShadow: "0 8px 24px rgba(15, 23, 42, 0.08)",
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: "20px",
  },
  title: {
    margin: 0,
    fontSize: "34px",
  },
  subtitle: {
    marginTop: "8px",
    color: "#6b7280",
  },
  statusPill: {
    background: "#e0f2fe",
    color: "#075985",
    padding: "10px 16px",
    borderRadius: "999px",
    fontWeight: "bold",
  },
  tabs: {
    display: "flex",
    gap: "10px",
    marginBottom: "20px",
  },
  tab: {
    padding: "12px 18px",
    border: "1px solid #d1d5db",
    background: "white",
    borderRadius: "999px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  activeTab: {
    padding: "12px 18px",
    border: "1px solid #2563eb",
    background: "#2563eb",
    color: "white",
    borderRadius: "999px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  card: {
    background: "white",
    padding: "28px",
    borderRadius: "18px",
    boxShadow: "0 8px 24px rgba(15, 23, 42, 0.08)",
  },
  cardHeader: {
    marginBottom: "20px",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
    gap: "18px",
  },
  label: {
    display: "block",
    fontWeight: "bold",
    marginBottom: "7px",
  },
  input: {
    width: "100%",
    padding: "12px",
    marginBottom: "16px",
    borderRadius: "10px",
    border: "1px solid #d1d5db",
    boxSizing: "border-box",
  },
  button: {
    padding: "13px 22px",
    background: "#2563eb",
    color: "white",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  disabledButton: {
    padding: "13px 22px",
    background: "#9ca3af",
    color: "white",
    border: "none",
    borderRadius: "10px",
    fontWeight: "bold",
  },
  secondaryButton: {
    marginTop: "18px",
    padding: "12px 18px",
    background: "#111827",
    color: "white",
    border: "none",
    borderRadius: "10px",
    cursor: "pointer",
    fontWeight: "bold",
  },
  successBox: {
    background: "#dcfce7",
    color: "#166534",
    padding: "14px",
    borderRadius: "12px",
    marginBottom: "15px",
    fontWeight: "bold",
  },
  errorBox: {
    background: "#fee2e2",
    color: "#991b1b",
    padding: "14px",
    borderRadius: "12px",
    marginBottom: "15px",
    fontWeight: "bold",
  },
  mutedText: {
    color: "#6b7280",
    marginTop: "6px",
  },
  loaderBox: {
    marginTop: "22px",
    display: "flex",
    alignItems: "center",
    gap: "12px",
    background: "#f9fafb",
    padding: "14px",
    borderRadius: "12px",
  },
  spinner: {
    width: "28px",
    height: "28px",
    border: "4px solid #d1d5db",
    borderTop: "4px solid #2563eb",
    borderRadius: "50%",
    animation: "spin 1s linear infinite",
  },
  previewWrapper: {
    marginTop: "15px",
    marginBottom: "20px",
    display: "flex",
    alignItems: "center",
    gap: "18px",
    background: "#f9fafb",
    padding: "15px",
    borderRadius: "14px",
  },
  previewImage: {
    width: "180px",
    height: "180px",
    objectFit: "cover",
    borderRadius: "14px",
    border: "1px solid #e5e7eb",
  },
  predictionCard: {
    marginTop: "22px",
    padding: "22px",
    background: "#eff6ff",
    border: "1px solid #bfdbfe",
    borderRadius: "14px",
  },
  emptyBox: {
    background: "#f9fafb",
    padding: "18px",
    borderRadius: "12px",
    color: "#6b7280",
  },
  summaryGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, minmax(0, 1fr))",
    gap: "14px",
  },
  metricCard: {
    background: "#f9fafb",
    padding: "16px",
    borderRadius: "14px",
    border: "1px solid #e5e7eb",
  },
  pathBox: {
    marginTop: "16px",
    background: "#f9fafb",
    padding: "14px",
    borderRadius: "12px",
    wordBreak: "break-all",
  },
  tableWrapper: {
    marginTop: "18px",
    overflowX: "auto",
  },
  table: {
    width: "100%",
    borderCollapse: "collapse",
  },
  th: {
    textAlign: "left",
    padding: "12px",
    background: "#f3f4f6",
    borderBottom: "1px solid #d1d5db",
  },
  td: {
    padding: "12px",
    borderBottom: "1px solid #e5e7eb",
  },
  smallText: {
    fontSize: "12px",
    wordBreak: "break-all",
  },
};

export default App;
