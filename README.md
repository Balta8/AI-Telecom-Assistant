# 🤖 AI Telecom Assistant

An intelligent customer support chatbot for telecommunications services, built with LangChain, Chainlit, and modern AI technologies. This assistant provides comprehensive support for telecom package information, recommendations, technical support, and frequently asked questions.

## ✨ Features

- 🎯 **Intelligent & Specialized**: Provides accurate answers to customer inquiries
- 📦 **Package Information**: Comprehensive details about all available service packages
- 🔍 **Smart Recommendations**: Suggests suitable packages based on customer needs
- 💬 **Technical Support**: Resolves technical and administrative issues
- 🧠 **Conversation Memory**: Remembers context throughout the conversation
- 🌐 **Modern Interface**: Beautiful and user-friendly Chainlit interface
- 🔍 **Vector Search**: ChromaDB-powered semantic search for accurate information retrieval
- 🛠️ **Modular Architecture**: Extensible node-based system for different functionalities

## 🏗️ Architecture

This project uses a modular, agent-based architecture:

- **Agent System**: LangChain-powered conversational agent
- **Node-based Tools**: Specialized nodes for different functionalities
- **Vector Database**: ChromaDB for semantic search and retrieval
- **Multiple Interfaces**: Chainlit and console-based interfaces
- **RAG Implementation**: Retrieval-Augmented Generation for accurate responses

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Git

### 1. Clone and Setup

```bash
# Clone the repository
git clone https://github.com/Balta8/AI-Telecom-Assistant.git
cd AI-Telecom-Assistant

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the root directory:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### 3. Initialize Vector Database

```bash
# Load data into ChromaDB
python3 -c "
from utils.chunking import NRowsChunker
from utils.ingest import ChromaIngestor
import json

# Load and process data
with open('data/data_improved.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

chunker = NRowsChunker(n=1)
ingestor = ChromaIngestor(chroma_dir='./chroma_store')
chunks = chunker.chunk(data)
ingestor.ingest(chunks)
print('✅ Data loaded successfully!')
"
```

### 4. Run the Application

#### Using Chainlit (Recommended)
```bash
chainlit run chainlit_app.py --port 8000
```

#### Using Console Interface
```bash
python3 app.py
```

#### Access the Application
- **Chainlit Interface**: `http://localhost:8000`
- **Console**: Direct terminal interaction

## 📁 Project Structure

```
AI-Telecom-Assistant/
├── 📄 agent.py                    # Main conversational agent
├── 📄 app.py                      # Console application
├── 📄 chainlit_app.py             # Chainlit web interface
├── 📄 config.py                   # Configuration settings
├── 📄 constants.py                # Application constants
├── 📄 requirements.txt            # Python dependencies
├── 📄 README.md                   # Project documentation
├── 📄 .gitignore                  # Git ignore rules
├── 📄 chainlit.md                 # Chainlit configuration
├── 📁 data/
│   ├── 📄 data.json              # Base telecom data
│   └── 📄 data_improved.json     # Enhanced data with features
├── 📁 src/nodes/                 # Specialized agent tools
│   ├── 📄 __init__.py
│   ├── 📄 faq_node.py           # FAQ handling
│   ├── 📄 package_info_node.py  # Package information
│   ├── 📄 package_recommendation_node.py  # Package recommendations
│   └── 📄 support_node.py       # Technical support
├── 📁 utils/                     # Utility modules
│   ├── 📄 chunking.py           # Data chunking strategies
│   ├── 📄 ingest.py             # Data ingestion to vector DB
│   └── 📄 retrievers.py         # Information retrieval
└── 📁 chroma_store/             # Vector database storage
```

## 🛠️ Available Tools

### 1. **Package Information Tool** (`package_info_tool`)
- Retrieves detailed information about specific telecom packages
- **Example**: "Tell me about Flex 70 package"

### 2. **Package Recommendation Tool** (`package_recommendation_tool`)
- Suggests suitable packages based on user requirements
- Lists all available packages
- **Example**: "I need a package for calls under 100 EGP"

### 3. **FAQ Tool** (`faq_tool`)
- Answers frequently asked questions
- **Example**: "How do I recharge my balance?"

### 4. **Support Tool** (`support_tool`)
- Provides technical and administrative support
- **Example**: "Router connection issues"

## ⚙️ Configuration

### Environment Variables
```bash
OPENAI_API_KEY=your_openai_api_key
```

### Application Settings (in `constants.py`)
```python
MAX_MESSAGE_LENGTH = 1000        # Maximum message length
MIN_MESSAGE_LENGTH = 2           # Minimum message length
MAX_ACTIVE_SESSIONS = 100        # Maximum active sessions
AGENT_MAX_ITERATIONS = 5         # Maximum agent iterations
```

## 🎯 Usage Examples

### Package Information Query
```
User: "What are the details of Flex 70 package?"
Assistant: "Package Name: Flex 70
Details: [Complete package information]
Price: [Package price]"
```

### Package Recommendation
```
User: "I need a package for calls under 100 EGP"
Assistant: "✅ Recommended Package: Flex 70
Why?
• Fits your budget
• Excellent call features
Price: 70 EGP"
```

### List All Packages
```
User: "What packages are available?"
Assistant: "📦 Flex Packages:
• Flex 70 — [Details] — Price: 70 EGP
• Flex 100 — [Details] — Price: 100 EGP

📦 Plus Packages:
• Plus 155 — [Details] — Price: 155 EGP"
```

## 🔧 Development

### Adding New Packages
1. Update `data/data_improved.json` with new package information
2. Reload the vector database:
```bash
python3 -c "
from utils.chunking import NRowsChunker
from utils.ingest import ChromaIngestor
import json

with open('data/data_improved.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

chunker = NRowsChunker(n=1)
ingestor = ChromaIngestor(chroma_dir='./chroma_store')
chunks = chunker.chunk(data)
ingestor.ingest(chunks)
print('✅ Data reloaded!')
"
```

### Adding New FAQs
1. Add questions to `data/data_improved.json` under the `faq` section
2. Reload the data using the script above

### Customizing Responses
Modify prompts in:
- `agent.py` - Main system prompts
- `src/nodes/` - Tool-specific prompts

### Extending Functionality
1. Create new node files in `src/nodes/`
2. Follow the existing pattern for tool creation
3. Register new tools in the agent configuration

## 🔍 Technologies Used

- **LangChain**: Agent framework and LLM orchestration
- **Chainlit**: Modern chat interface
- **ChromaDB**: Vector database for semantic search
- **OpenAI GPT**: Language model for conversations
- **Python**: Core programming language
- **JSON**: Data storage format

## 🐛 Troubleshooting

### Issue: "Package not found"
- Verify package exists in `data/data_improved.json`
- Reload the vector database
- Check package name spelling

### Issue: "API Error"
- Verify `OPENAI_API_KEY` is correct
- Check OpenAI account balance and usage limits
- Ensure internet connectivity

### Issue: "Agent doesn't remember context"
- Verify session management in `agent.py`
- Check conversation memory settings
- Restart the application

### Issue: "Slow responses"
- Check vector database size and chunking strategy
- Optimize retrieval parameters in `utils/retrievers.py`
- Consider using faster embedding models

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For support or inquiries:
- 📧 Email: [your-email@example.com]
- 🐛 Issues: [GitHub Issues](https://github.com/Balta8/AI-Telecom-Assistant/issues)
- 📖 Documentation: [Project Wiki](https://github.com/Balta8/AI-Telecom-Assistant/wiki)

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- Chainlit for the beautiful chat interface
- ChromaDB for the vector database solution
- OpenAI for the powerful language models

---

**Built with ❤️ for intelligent customer support**
