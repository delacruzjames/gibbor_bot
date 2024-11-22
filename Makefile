heroku_deploy:
	# Build the Docker image for the Heroku platform
	docker build --platform linux/amd64 -t registry.heroku.com/gibbor-bot/web .
	# Push the Docker image to the Heroku container registry
	docker push registry.heroku.com/gibbor-bot/web
	# Release the web process on Heroku
	heroku container:release web --app gibbor-bot
