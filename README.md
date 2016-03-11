## Implementation Details

Implementation was done in MATLAB using the VLFeat and Liblinear toolkits to aid recognition. The general flow of the program is as follows:

1. Process training image directory to get class folders
2. Iterate through each class folder using vl subset() to select random images to train on 3. Extract the DSIFT interest points from the training images
4. Cluster the descriptors using vl kmeans() for BOW representation
5. Use the BOW to build a kdtree using vl kdtreebuild()
6. Compute the approximated chi-square kernel using vl homkermap()
7. Use liblinear train() to train the SVM
8. Iterate through each of the test images
9. Extract each images DSIFT interest points, and then their kernel
10. Compare the test image kernel to the trained SVM to get class scores 11. Return the highest rating class score for each image


After running multiple tests, I found that the best number of images to train on was 20. Additionally, the best k for k-means clustering was 500.

To perform the classification on the test images, liblinear predict() was not used. Instead, I generated the histogram of each test image using the DSIFT interest points. A KDTree was used here as well to make it easier. After generating the historgram, this was used to generate the chi-square kernel for the image. The kernel is then compared to the weights returned by the liblinear SVM training (transposed) which returns the scores for each image class. The best score is then chosen by sorting and getting the top value, and this is the class clasification for that image.

Output of the program is in the form of a file, in a similar format to the provided groundtruth file. For more details, refer to the comments in the program. Also note that the efficiency of the program itself may not be great, as this is my first time using MATLAB. I found that using parfor loops greatly helped the speed, however this was one of the only optimisations I could find. Obviously, this does not affect classification accuracy though!

## Testing and Accuracy
The final classification accuracy of my program is ≈ 93%.
I ran 6 tests, modifying various parts of the program following the results from each test. The main things modified were the number of images used for training and k for clustering. The following shows a comparison of each tests result:

| | Test 1       | Test 2  | Test 3 | Test 4 | Test 5 | Test 6
| |------------- |---------|---------|---------|---------|---------
| Overall Accuracy | 86.23% | 89.65% | 92.49% | 93.40% | 93.17% | 93.74%
| # Incorrect Classifications | 121 | 91 | 66 | 58 | 60 | 57
| # Training Images | 5 | 10 | 15 | 25 | 25 | 20
| k | 500 | 500 | 500 | 750 | 1000 | 500
As we can see, the best result is in the final test using 20 training images per class and a k of 500. How- ever, as can be seen by comparing Tests 4 and 6, the improvement from the 5 image difference is minimal.
After determining these best values, I ran 3 more tests and averaged the classification accuracy to get the ≈ 93% figure above.

| Test | Accuracy |
|------|----------|
| Test 1 | 93.74% |
| Test 6.1 | 93.62% |
| Test 6.2 | 92.37% |
| Test 6.3 | 93.62% |
| **Avg** | **93.34%** |

It should also be noted that performing random number generation using the VLFeat’s vl twister() function prior to using vl phow() and train() increased the accuracy of each test by approximately 1%. This suggestion was taken from the VLFeat documentation. Note that this effect is not cumulative, as multiple uses of random number generation (using MATLAB’s rand() and randn() functions) did not improve classification accuracy (i.e. one was enough).

Below is the classification accuracy for each class during the final test (6). Worth noting is that the class to perform the worst in each test was the cup class. Changes to k and the number of test images did not improve performance for this category. The classification accuracy of the cup category stayed at 68.42% for the final 3 tests. This suggests that the upper bound on training images for the cup category is 20. This is perhaps due to the small sample size of the cup category in the training image directory.
| Class | Accuracy |
| ----- | -------- |
| airplanes | 97.38%
| anchor | 71.43%
| camera | 76.47%
| cup | 68.42%
| airplane | 82.35%
| emu | 72.22%
| grand piano | 100.00%
| headphone | 85.71%
| kangaroo | 89.66%
| lamp | 75.00%
| laptop | 96.30%
| lobster | 85.71%
| motorbike | 99.25%
| panda | 92.31%
| pizza | 94.44%
| rhino | 100.00%
| snoopy | 100.00%
| stapler | 80.00%
| wheelchair | 95.00%
| yin yang | 90.00%