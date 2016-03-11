function VisRecog()
%
% Implementation of visual recognition demo. General outline is as follows:
%   (1) Extract dense SIFT interest points
%   (2) Construct a visual dictionary based on training images
%   (3) Generate BOW features for all images
%   (4) Train SVM for object recognition based on training images
%   (5) Recognise objects in testing images
%
% Uses VLFeat for dense SIFT extraction and liblinear for SVM training
%
% Key values used: 
%     k (for k-means): 500
%     training images per class: 20
%     dsift steps: 2

% ======== | Global Vars | ======== %

trainDirectory = 'data/train';        % Directory holding training images
testDirectory = 'data/test';          % Directory holding testing images
outputFile = 'output.txt';            % Output file for predictions

numTrain = 20;                        % Number of images from each class to train on
numClasses = 20;                      % The number of classes for training

images = {};                          % Holds the images to train on
imageClass = {};                      % Status of trained images

% ======== | Program Setup | ======== %

% Get a listing of training directory contents
classes = dir(trainDirectory);
% Just get the names of the classes and ignore parent dirs
classes = { classes(3:numClasses+2).name };

% Iterate through the class directories
for ci = 1:length(classes)
  % Get all of the images in the class
  ims = dir(fullfile(trainDirectory, classes{ci}, '*.jpg'))';
  % Select the random images to train on
  ims = vl_colsubset(ims, numTrain);
  % Get just the relative names of the images
  ims = cellfun( @(x) fullfile(classes{ci}, x), {ims.name}, 'UniformOutput', false );
  % Append to the whole image array
  images = {images{:}, ims{:}};
  % Add to array holding image class training status
  imageClass{end + 1} = ci * ones(1, length(ims));
end

selTrain = find(mod(0:length(images) - 1, numTrain) < numTrain);
imageClass = cat(2, imageClass{:});

% Perform some random number generation prior to dsift extraction
vl_twister('state',1);

% ======== | Train Dictionary | ======== %

% Get some PHOW descriptors to train the dictionary
trainFeatures = vl_colsubset(selTrain, 30);
descriptors = {};

% The step option for phow extraction
phowOpts = {'Step', 2};

% Iterate through training image set
parfor i = 1:length(trainFeatures)
    % Get the image to train on
    im = imread(fullfile(trainDirectory, images{trainFeatures(i)}));
    % Convert image to single precision and resize if needed
    im = im2single(im);
    if size(im, 1) > 480
        im = imresize(im, [480 NaN]);
    end
    % Get the descriptors
    [~, descriptors{i}] = vl_phow(im, phowOpts{:}) ;
end

% Conctenate all of the descriptors and convert to single precision
descriptors = single( vl_colsubset(cat(2, descriptors{:}), 10e4) );

% Now cluster using kmeans to get words for BOW representation
words = vl_kmeans(descriptors, 500, 'algorithm', 'elkan', 'MaxNumIterations', 50);

% Build a kdtree for the BOW words
kdTree = vl_kdtreebuild(words);

% ======== | Generate BOW | ======== %

% Image histograms
histograms = {};

% Generate histrogram for each image
parfor i = 1:length(images)
    im = imread(fullfile(trainDirectory, images{i})) ;
    histograms{i} = processImage(im, phowOpts, words, kdTree);
end

histograms = cat(2, histograms{:});

% ======== | Training SVM | ======== %

% Compute approximated chi-square kernel
kernel = vl_homkermap(histograms, 1, 'kchi2', 'gamma', .5);

% Use liblinear to train the SVM
model = train(imageClass(selTrain)', sparse(double(kernel(:,selTrain))), '-s 3 -B 1.000000 -c 10.000000',  'col') ;

% ======== | Test! | ======== %

% Get all of the test images
testImDir = dir(testDirectory);
% Just get the names of the images and ignore parent dirs
testIms = { testImDir(3:size(testImDir)).name };

% Test image histograms
histograms = {};

labels = imageClass(selTrain)';

% Output file
fid = fopen(outputFile, 'wt');

% Process each image
for i = 1:length(testIms)
    im = imread(fullfile(testDirectory, testIms{i}));
    % Get the histogram for the test image
    histogram = processImage(im, phowOpts, words, kdTree);
    % Generate its kernel
    kernel = vl_homkermap(histogram, 1, 'kchi2', 'gamma', .5);
    % Compare with the model
    scores = (model.w(:,1:end-1)')' * kernel + (model.w(:,end)')';
    [~, best] = max(scores);
    imName = testIms{i};
    % Get the class suggestion
    class = classes{best};
    % Add to output file (image name, class)
    fprintf(fid, '%s %s\n', imName, class);
end

% Close output file
fclose(fid);

% ======== | Function: Process the image to get its descriptors | ======== %
function hist = processImage(im, phowOpts, words, kdTree)

% Convert image to single precision and resize if needed
im = im2single(im);
if size(im, 1) > 480
    im = imresize(im, [480 NaN]);
end
width = size(im,2);
height = size(im,1);
numWords = size(words, 2);

spatialX = [2, 4];
spatialY = [2, 4];

% Get the dsift features
[frames, descriptors] = vl_phow(im, phowOpts{:});

% Centroids
binsa = double(vl_kdtreequery(kdTree, words, single(descriptors), 'MaxComparisons', 50));

for i = 1:length(spatialX)
  binsx = vl_binsearch( linspace(1, width, spatialX(i) + 1), frames(1,:) );
  binsy = vl_binsearch( linspace(1, height, spatialY(i) + 1), frames(2,:) );

  % Combine bins
  bins = sub2ind([spatialY(i), spatialX(i), numWords],binsy, binsx, binsa);
  hist = zeros(spatialY(i) * spatialX(i) * numWords, 1);
  hist = vl_binsum(hist, ones(size(bins)), bins);
  hists{i} = single(hist / sum(hist));
end
hist = cat(1,hists{:});
% Result
hist = hist / sum(hist);