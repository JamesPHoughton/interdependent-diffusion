# Detective Game experiment
This folder contains the Empirica project for the detective game.


## Running the game server

To run this project locally, run the local server:

```sh
meteor
```

#### To run with a local version of the empirica core:
```sh
METEOR_PACKAGE_DIRS=/Library/MeteorPackages/ meteor
```

#### To inspect the server side code for debugging:
```sh
meteor --inspect
```

### To run experiment on galaxy server: (update your hostname)
1. uncomment lines in the `client: main.js` to make sure that they see the
production version
2. start the mongodb database on atlas
3. deploy
```sh
DEPLOY_HOSTNAME=us-east-1.galaxy-deploy.meteor.com meteor deploy detective.meteorapp.com --settings settings.json
```
4. run games
5. download data
6. shut down galaxy server
7. shut down mongodb database


Set up mongodb atlas with galaxy: https://www.okgrow.com/posts/mongodb-atlas-setup
troubleshooting `authentication fail` involved creating a new user with a simple password and trying again...

This game tends to exceed 100iops on the server side, so its best to provision one of the  
larger atlas servers, and ensure that it has a limit of 1000iops or so. m20 on
google's servers, might be the way to go?
