# csci0451-loupessis-finalproject
Parker Loupessis' Machine Learning Final Project 

## Project Proposal
### Group Members
Parker Loupessis

### Abstract
The problem this project aims to solve is that a lot of time is spent on breaking-down/labeling football practice and game film so that statistics and tendencies can be diagnosed. I'd like to create a model that could be a first step to speeding up this process by accurately labeling what route a wide receiver ran based on their (x,y) coordinates throughout the play. I plan to develop a model that uses a version of Logistic Regression (at least with deep learning) to classifiy each observation. I plan to evaluate the success of the model on how little the models accuracy changes as I add more and more possible labels.

### Motivation and Question
Important columns (gameId, playId, playDescription, nflId, PositionAbbr, x, y, and fram.id)
The dataset I have aquired contains the x, y coordinates of each player for each play of a single NFL game. Each player and play is distringuished by an ID number and important details about both are provided. Using this data, we can trace out the distinct path a player takes on the field. Our goal is to build a model that can accuarately classify the route that a WR ran on each play with only the x, y coordinates as the parameters. If we acheive high accuracy with the model we will add more route labels and try and predict the route wihtout seeing the whole thing. 


### Planned Deliverables
We plan to create a python package that contains all of the code for the model(s) that aim to accuarately classify the route that a WR ran on each play with only the x, y coordinates as the parameters. The first model will use route labels in, out, and go, and use the entirety of the play. Consecuative models will add more labels (e.g. slant, curl, corner, etc...) and/or use only the first only couple seconds of the play. We also plan to create a Jupyter notebook illustrating the use of the package to analyze data for each model.
Full success will mean the creation of mutiple models, as outlined above, that can take a sequence of the x, y coordinates of a WR on a single play and label the route they ran. While high accuracy would be ideal, it is more important that the model can perform the desired actions with the planned inputs. Partial success would mean having to change the model design to predict the WR's x, y coordinates (not labels) at the next defined time step with the previous x, y coordinates as the parameters. 

### Resources Required


### What You Will Learn

### Risk Statement

### Ethics Statement

### Tentative Timeline
