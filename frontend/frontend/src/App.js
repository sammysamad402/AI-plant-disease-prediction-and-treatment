import axios from "axios";
import { useEffect, useState } from "react";

function App() {
  const [file, setFile] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [imagePath, setImagePath] = useState(null);
  const [records, setRecords] = useState([]);

  const API_URL = "http://localhost:8000"; // backend URL

  const handleUpload = async () => {
    if (!file) return alert("Please select a file!");
    const formData = new FormData();
    formData.append("file", file);
    formData.append("device_id", "frontend_device");

    try {
      const res = await axios.post(`${API_URL}/detect`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      setPredictions(res.data.predictions);
      setImagePath(`${API_URL}/image/${res.data.image_path.split("/").pop()}`);
    } catch (err) {
      console.error(err);
      alert("Error uploading image");
    }
  };

  const fetchRecords = async () => {
    try {
      const res = await axios.get(`${API_URL}/records`);
      setRecords(res.data);
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchRecords();
  }, []);

  return (
    <div style={{ padding: "20px" }}>
      <h1>Agriculture Disease Detection</h1>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <button onClick={handleUpload}>Upload & Detect</button>

      {imagePath && (
        <div>
          <h2>Detection Result</h2>
          <img src={imagePath} alt="Detection" style={{ maxWidth: "400px" }} />
          <pre>{JSON.stringify(predictions, null, 2)}</pre>
        </div>
      )}

      <h2>Recent Records</h2>
      <ul>
        {records.map((rec) => (
          <li key={rec._id}>
            <b>{rec.device_id}</b> — {new Date(rec.timestamp * 1000).toLocaleString()}
          </li>
        ))}
      </ul>
    </div>
  );
}

export default App;
