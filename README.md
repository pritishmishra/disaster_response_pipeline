# Disaster Response Pipeline Project
This project analyses messages received during a disaster and tries to classify the messages into appropriate category. Any message can be annotated into multiple categories. This project builds a ML model using historical data of messages that had been pre-classified. 

Using this model, if any user provides a message as an input, the program will provide the categories for the message as the output.

e.g. Input: **this is a disaster call**
Output: This message could belong to these categories: **Related, Request, Aid-Related, Weather-Related, Floods, Direct Report**

# How is this project organized:
- The experiment folder is mainly for the developers. This contains Jupyter files which were used to develop the code for this project. If you want to tinker with the code or see some quick results/visualizations, head over to this folder. In order to use, it is recommended to have a Conda setup and Jupypter notebook.

- The workspace contains the main and final code for this project. Head over to the workspace folder for more details on how to use this.

# How to use this project:
- Currently, the project already contains the code for training a ML model and showcasing the visualisatons in form of a Flash web-app.
- You can provide a messages and categories csv files to recreate and re-train the model.
- Once the model is trained, you can run the Flask web-app to see the results.
- This Flask web-app can also exported and used by an user to find the categories of any new message.
- More details on how to re-create/re-train the model or how to run the Flask web-app is present [here](https://github.com/pritishmishra/disaster_response_pipeline/blob/master/workspace/README.md)
