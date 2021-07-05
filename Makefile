# Interact Variables
# defaults
CHECKPOINT_DIR = "counselchat-convai"
DATA = "data/counsel_chat_250-tokens_full.json"
TEMPERATURE = 0.7
MAX_LENGTH = 30
MAX_HISTORY = 0

# HELP
# This will output the help for each task
# thanks to https://marmelab.com/blog/2016/02/29/auto-documented-makefile.html
.PHONY: help

help: ## This help.
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

.DEFAULT_GOAL := help


# DOCKER TASKS
# Build the container
build:
	docker system prune && \
	docker build -t convai -f docker/Dockerfile .

## Interact with the model
interact:
	docker run -it --rm -v $(shell pwd):$(shell pwd) convai:latest \
	/bin/sh -c 'cd $(shell pwd); \
	python3 interact.py --model_checkpoint $(CHECKPOINT_DIR) --temperature $(TEMPERATURE) --max_length $(MAX_LENGTH) --max_history $(MAX_HISTORY) --dataset_path $(DATA); \
	'