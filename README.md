# Final Project - Visual Analytics
Author: Christian Norup Kolding (201805727)
<br>
Date: 16th of June 2023


# Description
In this last assignment for my portfolio the goal is to train a a classifier on a dataset consisting of 227 x-ray images of "normal" lungs versus lungs infected with Covid-19. My motivation for choosing this particular topic is derived from my interest in human longevity and health. 

# Methods
I fed the VGG16 classifier with the 227 images, splitting these into a training and test set respectively. Then the data was fed to a pretrained model, from which I was able to create a classification report. Furthermore, I made a plot showing the loss and accuracy curve. Both the report and the plot is saved under ```out```.

# Results
The classification report shows that the classifier has a high level of fitness since it is well equipped to identify both "normal" lungs and lungs infected with Covid-19, respectively. This may be due to the fact that x-ray of the human body images of are used specifically for the purpurse of identifying anomalies, and therefore a relatively small dataset is enough for the model to "learn".

The loss and accuracy curve plot shows that the model gets better at identifying the two classes over time. This because the gap between the "training loss" and "validation loss" lines deminish over time. 

