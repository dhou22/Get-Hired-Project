# Get Hired Project - AI Recruitement Assistant
-----
<img width="1024" height="559" alt="image" src="https://github.com/user-attachments/assets/21b6f1a9-2a72-422e-9b45-c56c532484d6" />

-----

A machine learning project that leverages HuggingFace transformers and Weaviate vector database to create semantic embeddings of resumes for intelligent matching and retrieval.

## 📋 Overview

This project implements a resume processing pipeline that:
- Generates semantic embeddings from resume text using HuggingFace models
- Stores embeddings in Weaviate vector database for efficient similarity search
- Enables semantic search and matching of resumes based on job requirements

## 🚀 Features

- **Semantic Understanding**: Uses transformer-based models to capture deep semantic meaning from resume text
- **Vector Storage**: Efficient storage and retrieval using Weaviate's vector database
- **Similarity Search**: Find best-matching resumes based on job descriptions or requirements
- **Scalable Architecture**: Built to handle large volumes of resume data

## 🛠️ Technologies Used

- **Python**: Core programming language
- **HuggingFace Transformers**: Pre-trained models for text embeddings
- **Weaviate**: Vector database for storing and querying embeddings
- **Jupyter Notebook**: Interactive development environment

## 📦 Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook
- Docker (for Weaviate)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/dhou22/Get-Hired-Project.git
cd Get-Hired-Project
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install individually:
```bash
pip install transformers torch weaviate-client sentence-transformers pandas numpy
```

3. Start Weaviate instance:
```bash
docker run -d \
  -p 8080:8080 \
  -e AUTHENTICATION_ANONYMOUS_ACCESS_ENABLED=true \
  -e PERSISTENCE_DATA_PATH='/var/lib/weaviate' \
  semitechnologies/weaviate:latest
```

## 📓 Usage

### Running the Notebook

1. Launch Jupyter Notebook:
```bash
jupyter notebook resume-embedding-huggingface-weaviate-storage.ipynb
```

2. Follow the notebook cells sequentially to:
   - Load and preprocess resume data
   - Generate embeddings using HuggingFace models
   - Store embeddings in Weaviate
   - Query and retrieve similar resumes

### Basic Example

```python
from transformers import AutoTokenizer, AutoModel
import weaviate

# Initialize HuggingFace model
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Connect to Weaviate
client = weaviate.Client("http://localhost:8080")

# Generate embedding for resume text
resume_text = "Your resume text here..."
# ... (embedding generation code)

# Store in Weaviate
# ... (storage code)

# Query for similar resumes
query = "Software engineer with Python experience"
# ... (query code)
```

## 🏗️ Project Structure

```
Get-Hired-Project/
├── resume-embedding-huggingface-weaviate-storage.ipynb  # Main notebook
├── README.md                                             # This file
├── requirements.txt                                      # Python dependencies
└── data/                                                 # (Optional) Sample data directory
```

## 🔍 How It Works

1. **Text Preprocessing**: Resume text is cleaned and formatted
2. **Embedding Generation**: HuggingFace transformer models convert text into dense vector representations
3. **Vector Storage**: Embeddings are stored in Weaviate with metadata
4. **Semantic Search**: Queries are embedded and compared using cosine similarity
5. **Result Ranking**: Most similar resumes are retrieved and ranked

## 🎯 Use Cases

- **Recruitment**: Match candidates to job openings
- **Talent Search**: Find candidates with specific skills
- **Resume Screening**: Automate initial resume filtering
- **Career Services**: Help job seekers find relevant opportunities

## 📊 Performance

The system's performance depends on:
- Quality and size of the embedding model
- Number of resumes in the database
- Query complexity

Typical query times: < 100ms for databases with thousands of resumes

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👤 Author

**dhou22**
- GitHub: [@dhou22](https://github.com/dhou22)

## 🙏 Acknowledgments

- HuggingFace for transformer models
- Weaviate for vector database technology
- The open-source community

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you find this project helpful, please consider giving it a star!
