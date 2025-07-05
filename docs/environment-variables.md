# Environment Variables Documentation

## Required Variables

### API Keys
- `OPENAI_API_KEY`: Your OpenAI API key (format: sk-...)
- `ANTHROPIC_API_KEY`: Your Anthropic Claude API key (format: sk-ant-...)

## Optional Variables

### Additional LLM Providers
- `HUGGINGFACE_API_KEY`: HuggingFace Inference API key (format: hf_...)
- `COHERE_API_KEY`: Cohere API key
- `AI21_API_KEY`: AI21 Labs API key
- `GOOGLE_AI_API_KEY`: Google AI API key
- `AZURE_OPENAI_API_KEY`: Azure OpenAI API key
- `AZURE_OPENAI_ENDPOINT`: Azure OpenAI endpoint URL
- `REPLICATE_API_KEY`: Replicate API key
- `TOGETHER_API_KEY`: Together AI API key

### Backup Configuration
- `GOOGLE_DRIVE_FOLDER_ID`: Google Drive folder ID for backups
- `BACKUP_RETENTION_DAYS`: Number of days to keep backups (default: 180)
- `GOOGLE_APPLICATION_CREDENTIALS`: Path to Google service account JSON

### Application Settings
- `ENVIRONMENT`: Application environment (development/production)
- `LOG_LEVEL`: Logging level (debug/info/warning/error)
- `DEBUG`: Enable debug mode (true/false)

### Service Configuration
- `REDIS_URL`: Redis connection URL (default: redis://redis:6379)
- `OLLAMA_HOST`: Ollama server URL (default: http://ollama:11434)

### Security (Production)
- `SECRET_KEY`: Application secret key (change in production)
- `JWT_SECRET`: JWT signing secret (change in production)

## Example .env File

```bash
# Required
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-claude-key-here

# Optional
HUGGINGFACE_API_KEY=hf_your-huggingface-key-here
GOOGLE_DRIVE_FOLDER_ID=your-google-drive-folder-id

# Application
ENVIRONMENT=development
LOG_LEVEL=info
DEBUG=true

# Services
REDIS_URL=redis://redis:6379
OLLAMA_HOST=http://ollama:11434

# Security (change in production)
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret-here
```