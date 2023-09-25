Since the frask app.py looks for a container called "model-container" you need to run it with that name. Also specifying port 5000. Finally the image to load is called flaskap-container.
This is the command to run the docker container to be ready for the input / output in JSON format.

docker run --name model-container -p 5000:5000 -d flaskap-container

Don't forget to start the docker daemon! 
On my system WSL it is:

sudo service docker start