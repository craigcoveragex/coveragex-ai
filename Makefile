.PHONY: help setup start stop restart logs status test clean backup

# Colors
BLUE=\033[0;34m
GREEN=\033[0;32m
RED=\033[0;31m
YELLOW=\033[1;33m
NC=\033[0m

help: ## Show this help message
	@echo "$(BLUE)🧠 AI Dev Superstack - Intelligent Routing Edition$(NC)"
	@echo "$(GREEN)============================================$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

setup: ## 🚀 Initial setup and configuration
	@echo "$(BLUE)🚀 Setting up AI Dev Superstack...$(NC)"
	@chmod +x setup.sh
	@./setup.sh

start: ## ▶️ Start all services
	@echo "$(BLUE)▶️ Starting services...$(NC)"
	@docker-compose up -d
	@echo "$(GREEN)✅ Services started!$(NC)"
	@make status

stop: ## ⏹️ Stop all services
	@echo "$(BLUE)⏹️ Stopping services...$(NC)"
	@docker-compose down
	@echo "$(GREEN)✅ Services stopped!$(NC)"

restart: ## 🔄 Restart all services
	@echo "$(BLUE)🔄 Restarting services...$(NC)"
	@docker-compose restart
	@echo "$(GREEN)✅ Services restarted!$(NC)"

logs: ## 📋 Show logs from all services
	@docker-compose logs -f --tail=100

logs-api: ## 📋 Show FastAPI logs
	@docker-compose logs -f fastapi --tail=100

status: ## 📊 Show service status
	@echo "$(BLUE)📊 Service Status:$(NC)"
	@docker-compose ps
	@echo ""
	@echo "$(BLUE)🔍 Health Checks:$(NC)"
	@curl -s http://localhost:8000/health | python3 -m json.tool || echo "$(RED)❌ FastAPI not responding$(NC)"

test: ## 🧪 Run a test request
	@echo "$(BLUE)🧪 Testing intelligent routing...$(NC)"
	@curl -s -X POST http://localhost:8000/api/v1/chat \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Write a Python function to calculate fibonacci numbers"}' \
		| python3 -m json.tool

routing-stats: ## 📊 Show routing statistics
	@echo "$(BLUE)📊 Routing Statistics:$(NC)"
	@curl -s http://localhost:8000/api/v1/routing/stats | python3 -m json.tool

benchmarks: ## 📊 Show model benchmarks
	@echo "$(BLUE)📊 Model Benchmarks:$(NC)"
	@curl -s http://localhost:8000/api/v1/models/benchmarks | python3 -m json.tool

explain: ## 🤔 Explain routing decision for a prompt
	@echo "$(BLUE)🤔 Routing Explanation:$(NC)"
	@curl -s -X POST http://localhost:8000/api/v1/routing/explain \
		-H "Content-Type: application/json" \
		-d '{"prompt": "Write a Python function to sort data"}' \
		| python3 -m json.tool

models: ## 🧠 List available models
	@echo "$(BLUE)🧠 Ollama Models:$(NC)"
	@docker-compose exec ollama ollama list

pull-model: ## 📥 Pull an Ollama model (usage: make pull-model MODEL=llama3)
	@echo "$(BLUE)📥 Pulling model: $(MODEL)$(NC)"
	@docker-compose exec ollama ollama pull $(MODEL)

clean: ## 🧹 Clean up containers and volumes
	@echo "$(RED)🧹 Cleaning up...$(NC)"
	@docker-compose down -v
	@echo "$(GREEN)✅ Cleanup complete!$(NC)"

backup: ## 💾 Create manual backup
	@echo "$(BLUE)💾 Creating backup...$(NC)"
	@docker-compose exec backup-service python3 /app/backup_to_drive.py
	@echo "$(GREEN)✅ Backup complete!$(NC)"