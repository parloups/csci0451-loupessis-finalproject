#csci0451-loupessis-finalproject
Parker Loupessis’ Machine Learning Final Project

##Project Proposal

###Group Members
Parker Loupessis

###Abstract
The problem this project aims to solve is that a lot of time is spent on breaking down/labeling football practice and game film so that statistics and tendencies can be diagnosed. I’d like to create a model that could be a first step toward speeding up this process by accurately labeling the route a wide receiver ran based on their x, y coordinates throughout the play. I plan to develop a model that uses a version of Logistic Regression (at least with deep learning) to classify each observation. I plan to evaluate the model's success by measuring how little its accuracy changes as I add more possible labels.

###Motivation and Question
The dataset I have acquired contains the x, y coordinates of each player for each play of a single NFL game. Each player and play is distinguished by an ID number, and important details about both are provided. Using this data, we can trace out the distinct path a player takes on the field. Our goal is to build a model that can accurately classify the route that a WR ran on each play with only the x, y coordinates as the parameters. If we achieve high accuracy with the model, we will add more route labels and try to predict the route without seeing the whole thing.
Important columns (gameId, playId, playDescription, nflId, PositionAbbr, x, y, and fram.id)

###Planned Deliverables
We plan to create a Python package that contains all of the code for the model(s) that aim to accurately classify the route that a WR ran on each play, with only the x, y coordinates as the parameters. The first model will use route labels in, out, and go, and use the entirety of the play. Consecutive models will add more labels (e.g., slant, curl, corner, etc) and/or use only the first only couple of seconds of the play. We also plan to create a Jupyter notebook illustrating the use of the package to analyze data for each model.
Full success will mean the creation of multiple models, as outlined above, that can take a sequence of the x, y coordinates of a WR on a single play and label the route they ran. While high accuracy would be ideal, it is more important that the model can perform the desired actions with the planned inputs. Partial success would mean having to change the model design to predict the WR’s x, y coordinates (not labels) at the next defined time step with the previous x, y coordinates as the parameters.

###Resources Required
For this project, we will be using the 2019 NFL Big Data Bowl dataset. The available dataset contains the x, y coordinates from one game (09/07/2017, Kansas City Chiefs @ New England Patriots) of each player for each play. We will filter it to only look at receivers on passing plays. We will create the targets by watching the game film (NFL+ Subscription needed) and creating the labels for each route run.

###What You Will Learn
Through this project, I am taking my first stab at using ML to solve a problem related to football. As I hope to work in the football sphere in the future, I hope to learn crucial lessons about how I can use ML to support myself as well as others. One of the biggest things I expect to learn is how to work with large, messy(ish) datasets that will require a lot of data prepping to achieve the goal. I also anticipate that creating my own targets might be difficult, but I imagine that it is something I am going to have to do a lot of if I intend to work in football.

###Risk Statement
One thing that could prevent us from achieving the full deliverables is the formulation of the labels. I expect this to be rather time-consuming and, depending on whether we can access the game film or the broadcast footage, rather difficult. If no film or footage can be found, we can make a plot of each route and try to label from that, but this would be the least reliable way of creating the targets. Another thing that could prevent us from achieving the full deliverable is if the model struggles as we add more target labels. There might be too little difference between some of the different routes for the model to distinguish, and that could hurt overall performance.

###Ethics Statement
If our project were successful and deployed, it would show that we could use ML to automate (at least some of) the process of breaking down football film. This would help coaches, scouts, and assistants a lot, as breaking down a game is time-consuming (and most don’t find it that fun). Anyone working in the football sphere has the potential to benefit from this because they could spend more time actually analyzing the film and the stats, not just prepping to do so. Some of these same people could also be harmed by it, as it could mean less of a demand for lower-ranked coaches, scouts, or assistants. Then, if fewer are needed, it makes it harder for someone to break into the field and work their way up the ranks. The world will probably be about the same after this project because it will only affect/help a relatively small subset of people, and while it will likely help the jobs of some, it might also keep others from getting jobs.

###Tentative Timeline
Two-week goal: have the data prepped and ready. Filter the data we plan to actually use and create all of the targets.
Four-week goal: build and implement the models. Have a notebook demonstrating the use of the models.
