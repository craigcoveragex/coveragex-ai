#!/bin/bash
set -e

echo "🧠 AI Dev Superstack - Intelligent Routing Setup"
echo "=============================================="

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo -e "${RED}❌ Docker Compose is not installed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker and Docker Compose detected${NC}"

# Create directory structure
echo -e "${BLUE}📁 Creating directory structure...${NC}"
mkdir -p fastapi/{services,models,routers,config}
mkdir -p data/conversations
mkdir -p credentials
mkdir -p monitoring
mkdir -p scripts

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo -e "${BLUE}🔧 Creating .env file...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️ Please edit .env file with your API keys${NC}"
    echo -e "${YELLOW}   Required: OPENAI_API_KEY, ANTHROPIC_API_KEY${NC}"
    echo -e "${YELLOW}   Optional: HUGGINGFACE_API_KEY, GOOGLE_DRIVE_FOLDER_ID${NC}"
else
    echo -e "${GREEN}✅ .env file exists${NC}"
fi

# Check for Google Drive credentials
if [ ! -f credentials/service-account.json ]; then
    echo -e "${YELLOW}ℹ️ Google Drive backup: Add service account JSON to credentials/service-account.json${NC}"
fi

# Create monitoring config
cat > monitoring/prometheus.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'fastapi'
    static_configs:
      - targets: ['fastapi:8000']
    metrics_path: '/metrics'
    scrape_interval: 10s
EOF

# Pull images
echo -e "${BLUE}📥 Pulling Docker images...${NC}"
if ! docker-compose pull; then
    echo -e "${YELLOW}⚠️ Some images failed to pull, continuing...${NC}"
fi

# Start services
echo -e "${BLUE}🚀 Starting services...${NC}"
if ! docker-compose up -d; then
    echo -e "${RED}❌ Failed to start services${NC}"
    exit 1
fi

# Wait for services
echo -e "${BLUE}⏳ Waiting for services to start...${NC}"
sleep 15

# Check if Ollama is ready before pulling models
echo -e "${BLUE}📥 Setting up Ollama models...${NC}"
for i in {1..30}; do
    if docker-compose exec -T ollama ollama list &>/dev/null; then
        echo -e "${GREEN}✅ Ollama is ready${NC}"
        break
    fi
    echo -e "${YELLOW}⏳ Waiting for Ollama... ($i/30)${NC}"
    sleep 2
done

# Pull models with error handling
docker-compose exec -T ollama ollama pull llama3 || echo -e "${YELLOW}⚠️ Failed to pull llama3${NC}" &
docker-compose exec -T ollama ollama pull mistral || echo -e "${YELLOW}⚠️ Failed to pull mistral${NC}" &
wait

# Final message
echo -e "${GREEN}✅ AI Dev Superstack is ready!${NC}"
echo ""
echo -e "${BLUE}🌐 Access your services:${NC}"
echo -e "  API Docs: ${GREEN}http://localhost:8000/docs${NC}"
echo -e "  Health: ${GREEN}http://localhost:8000/health${NC}"
echo -e "  Prometheus: ${GREEN}http://localhost:9090${NC}"
echo ""
echo -e "${BLUE}📊 View routing stats:${NC}"
echo -e "  ${YELLOW}make routing-stats${NC}"
echo ""
echo -e "${BLUE}🧪 Test the system:${NC}"
echo -e "  ${YELLOW}make test${NC}"
echo ""
echo -e "${YELLOW}⚠️ Next steps:${NC}"
echo -e "  1. Add your API keys to .env file"
echo -e "  2. (Optional) Add Google Drive service account for backups"
echo -e "  3. Test different prompts to see intelligent routing in action"