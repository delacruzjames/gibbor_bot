# Define variables
DOCKER_IMAGE_NAME=registry.heroku.com/gibbor-bot/web
HEROKU_APP_NAME=gibbor-bot
DOCKER_CONTAINER_NAME=fastapi_app

# Target: Build Docker image for the Heroku platform
build:
	@echo "Building Docker image for Heroku..."
	docker build --platform linux/amd64 -t $(DOCKER_IMAGE_NAME) .

# Target: Push Docker image to Heroku container registry
push:
	@echo "Pushing Docker image to Heroku..."
	docker push $(DOCKER_IMAGE_NAME)

# Target: Release the web process on Heroku
release:
	@echo "Releasing web process on Heroku..."
	heroku container:release web --app $(HEROKU_APP_NAME)

# Target: Full deploy process
deploy: build push release
	@echo "Deployment to Heroku complete!"

# Target: Access the Docker container with bash
bash:
	@echo "Accessing the Docker container..."
	docker exec -it $(DOCKER_CONTAINER_NAME) bash

.PHONY: build push release deploy bash
