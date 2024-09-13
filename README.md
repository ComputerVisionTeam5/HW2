# HW2
Repository for Homework 2.

For the second homework assigment:

1) Implement your own version of region growing algorithm:

    - You should input a seed.
    - The segmentation should be able to extract a full object from the image

2) Implement the following clustering algorithms tailoring them for color images:

    - K-means algorithm
    - Fuzzy -c means Algorithm
    - Both should work on standard color images and
        - Their output should be a new image where each pixel has the mean color of the class.

3) Implement your own version of the outsu algorithm

    - Extract the boundaries of the object

4) For every single output of the avove results:

    - Clean segmentation errors using Opening
    - Clean Segmentation errors using Clossing

5) For the binary image created by Otsu threshold:

    - Compute the distance map to all the object boudaries 

6) Implement your own version of the Skeleton Algorithm and apply to the Otsu Binary Image

7) Optical Flow: Do the tutorial: https://learnopencv.com/optical-flow-in-opencv/

8) Implement a Segmentation pipeline that does the follwing:

    - Do a unsupervised segmentation of a color image
    - Extract the connected components
    - Compute basic features for each extracted component:
        - Area, perimeter and Hu moments
        - Mean, variance of each color
