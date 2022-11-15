# CMPSC 448 - Homework 4
## Chase Rensberger - chr5119

To whoever is grading this: this assignment has a lot of files so I made this README to explain where everything is. 

- Problem 1 is located in Problem1.pdf
- Problem 2 is contained in the Jupyter Notebook called Problem2.ipynb (This notebook also references/uses 3 other python files called Problem2BoostedDecisionTrees.py; Problem2RandomForests.py; Problem2SVMwGK.py)
- Problem 3 is located in Problem3.py and relevant plots for problem 3 are located in Problem3Plots.pdf with comments inside of the python file to explain what is going on

Also because I didn't want to put my explanation for #3 exclusively in the comments, I'll put my answers here as well:

We can see that k=5 minimized our objective function in the range 1-5 so we choose that to be our number of clusters. With this knowledge we can create a plot showing how objective changes with number of iterations (plots 3-7 in the pdf) and create a plot with the data colored by assignment, and the cluster centers (Plot 8 in the pdf). All of these plots are avaliable in Problem3Plots.pdf. 

You can also look through the comments for the origin of how the plots in the PDF were generated. There will be a label above the code used to generate that plot. Example: "# Used for plot 1 in the pdf"

Also, in the final plot (plot 8); the red dots are the centroids/cluster centers.