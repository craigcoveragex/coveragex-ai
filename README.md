# 🧠 AI Dev Superstack - Intelligent Routing Edition

A development and testing environment for multi-LLM applications with performance-based intelligent routing.

## 🚀 Features

### Core Features (Implemented)
- **🎯 Intelligent Routing**: Automatically routes requests to the best-performing model based on:
  - Task type classification (coding, analysis, creative, etc.)
  - Historical performance metrics
  - Model benchmark scores
- **📊 Performance Tracking**: Real-time tracking of model performance
- **🔄 Multi-Provider Support**: OpenAI, Claude, HuggingFace, and local Ollama models
- **💾 Google Drive Backup**: Automated backups with 180-day retention
- **📈 Monitoring**: Prometheus metrics for all services

### Supported Providers
- ✅ **OpenAI** (GPT-4, GPT-3.5)
- ✅ **Anthropic** (Claude)
- ✅ **HuggingFace** (Open source models)
- ✅ **Ollama** (Local models: Llama3, Mistral)
- 🔜 **Cohere** (Ready to enable)
- 🔜 **AI21 Labs** (Ready to enable)
- 🔜 **Google AI** (Ready to enable)
- 🔜 **Azure OpenAI** (Ready to enable)
- 🔜 **Replicate** (Ready to enable)
- 🔜 **Together AI** (Ready to enable)

## 📋 Requirements

- Docker & Docker Compose
- 32GB RAM (for local models)
- API keys for OpenAI and Claude (required)
- HuggingFace API key (optional)
- Google Cloud service account (optional, for backups)

## ⚡ Quick Start

1. **Clone and setup**
   ```bash
   git clone <repository>
   cd ai-dev-superstack
   make setup
   ```

2. **Configure API keys**
   ```bash
   nano .env
   # Add your OPENAI_API_KEY and ANTHROPIC_API_KEY
   ```

3. **Start the system**
   ```bash
   make start
   ```

4. **Test intelligent routing**
   ```bash
   make test
   ```

## 🎯 How Intelligent Routing Works

The system analyzes each request and routes it to the best model based on:

1. **Task Classification**: Identifies the type of task (coding, analysis, creative, etc.)
2. **Performance Metrics**: Considers real-time success rates and response times
3. **Benchmark Scores**: Uses pre-configured benchmark scores for each model
4. **Availability**: Routes only to available services

### Example Routing Decisions

- **Coding Task** → Likely routes to OpenAI (GPT-4) or Claude
- **Creative Writing** → May choose Claude or HuggingFace
- **Simple Q&A** → Could use Ollama for cost efficiency
- **Complex Analysis** → Typically selects Claude for best reasoning

## 📊 Monitoring Performance

View routing statistics:
```bash
make routing-stats
```

View model benchmarks:
```bash
make benchmarks
```

Explain routing decisions:
```bash
make explain
```

Or test with custom prompt:
```bash
curl -X POST http://localhost:8000/api/v1/routing/explain \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Your test prompt here"}'
```

## 🛠️ Development Commands

```bash
make help           # Show all commands
make start          # Start services
make stop           # Stop services
make logs           # View logs
make status         # Check service health
make test           # Run test request
make routing-stats  # View routing statistics
make benchmarks     # View model benchmarks
make explain        # Explain routing decision for a prompt
make backup         # Create manual backup
```

## 📁 Project Structure

```
ai-dev-superstack/
├── fastapi/           # Main API application
│   ├── services/      # LLM service integrations
│   ├── routers/       # Intelligent routing logic
│   └── config/        # Model benchmarks
├── data/              # Conversation storage
├── credentials/       # Service account keys
└── monitoring/        # Prometheus config
```

## 🔧 Configuration

### Environment Variables

- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `ANTHROPIC_API_KEY`: Your Claude API key (required)
- `HUGGINGFACE_API_KEY`: HuggingFace API key (optional)
- `GOOGLE_DRIVE_FOLDER_ID`: Folder ID for backups (optional)

### Adding New Providers

Uncomment the desired provider in `.env`:
```bash
# COHERE_API_KEY=your-key-here
# AI21_API_KEY=your-key-here
```

## 📈 Performance Optimization

The system is optimized for development/testing with:
- Resource limits on containers
- Efficient caching with Redis
- Smart model selection to minimize costs
- Local models for frequent testing

## 🆘 Troubleshooting

**Services not starting?**
```bash
make status
make logs
```

**Models not routing correctly?**
```bash
make routing-stats  # Check performance metrics
make benchmarks     # View model scores
```

**Need more models?**
```bash
make pull-model MODEL=codellama
```

## 📄 License

MIT License - See LICENSE file for details

---

Built with ❤️ for efficient multi-LLM development and testing