heroku_deploy:
    docker build --platform linux/amd64 -t registry.heroku.com/gibbor-bot/web .
    docker push registry.heroku.com/gibbor-bot/web
    heroku container:release web --app gibbor-bot