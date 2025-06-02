cellSize = [8 8];

% Define the full path to the .mat file
modelPath = 'C:\Users\Asus\Downloads\YOLO Result\svmClassifier.mat';

% Load the trained model from the file
load(modelPath, 'svmClassifier');

% Define the input and output directories
inputImagePath = 'C:\Users\Asus\Downloads\YOLO Result\Real-time image detection\input.jpg';  % Input image path
outputImagePath = 'C:\Users\Asus\Downloads\YOLO Result\Real-time image detection\outputmatlab.jpg';  % Output image path

% Define TCP/IP client
serverIP = '192.168.166.19';  % Replace with your server's IP address
serverPort = 6000;      % Replace with your server's port number
client = tcpclient(serverIP, serverPort);
    
% Create a figure window and hold the handle
hFig = figure('Name', 'Real-Time Image Detection and Classification', 'NumberTitle', 'off');
hAx = axes('Parent', hFig);

while ishandle(hFig)
    % Read the input image
    img = imread(inputImagePath);
    
    % Convert the image to RGB if needed
    if size(img, 3) == 1
        img = cat(3, img, img, img);
    end
    
    % Convert the image to grayscale for processing
    grayImg = rgb2gray(img);
    
    % Resize image to a standard size (same as during training)
    grayImg = imresize(grayImg, [128 128]);
    
    % Apply histogram equalization for better contrast
    grayImg = histeq(grayImg);
    
    % Extract features
    hogFeatures = extractHOGFeatures(grayImg, 'CellSize', cellSize);
    lbpFeatures = extractLBPFeatures(grayImg);
    houghFeatures = extractHoughFeatures(grayImg);
    edgeFeatures = extractEdgeFeatures(grayImg);
    gaborFeatures = extractGaborFeatures(grayImg);
    colorHist = extractColorHistogram(img); % Use the original RGB image
    
    % Combine features
    combinedFeatures = [hogFeatures, lbpFeatures, houghFeatures, edgeFeatures, gaborFeatures, colorHist];
    
    % Normalize features
    combinedFeatures = normalize(combinedFeatures);
    
    % Predict the label of the current image
    predictedLabel = predict(svmClassifier, combinedFeatures);
    
    % Display the predicted label on the image
    imgWithLabel = insertText(img, [10, 10], char(predictedLabel), 'FontSize', 18, 'BoxColor', 'red', 'BoxOpacity', 0.6);
    
    % Save the image with the predicted label to the output directory
    imwrite(imgWithLabel, outputImagePath);
    
    % Display the image
    if ishandle(hFig)
        imshow(imgWithLabel, 'Parent', hAx); % Use hAx to specify the current axes
        title(hAx, ['Predicted Label: ', char(predictedLabel)]);
        drawnow; % Ensure the GUI updates
    end
    
    % Check prediction and send data to TCP/IP server
    if strcmp(predictedLabel, 'ModerateAccident') || strcmp(predictedLabel, 'SevereAccident')
        % Send '1' to the server
        write(client, '1', 'char');
        
        % Wait for 2 seconds
        pause(2);
        
        % Send '0' to the server
        write(client, '0', 'char');
    end
    
    % Pause for a short duration to allow image to update
    pause(0.1);  % Adjust the pause duration as needed (e.g., 1 second)
end

% Close the TCP/IP client connection
%clear client;

%% Function to extract Hough Transform features from an image

function houghFeatures = extractHoughFeatures(img)
    % Detect edges using the Canny edge detector
    edges = edge(img, 'canny');
    
    % Perform Hough transform to find lines
    [H, T, R] = hough(edges);
    
    % Find peaks in the Hough transform matrix
    peaks = houghpeaks(H, 5, 'threshold', ceil(0.3 * max(H(:))));
    
    % Extract lines based on Hough transform
    lines = houghlines(edges, T, R, peaks);
    
    % Calculate some basic statistics about the lines
    numLines = length(lines);
    lineLengths = arrayfun(@(line) norm(line.point1 - line.point2), lines);
    
    % Aggregate features (e.g., number of lines, mean line length, max line length)
    houghFeatures = [numLines, mean(lineLengths), max(lineLengths)];
    
    % If there are no lines detected, fill with zeros
    if isempty(lineLengths)
        houghFeatures = [0, 0, 0];
    end
end

%% Function to extract Edge features from an image
function edgeFeatures = extractEdgeFeatures(img)
    % Detect edges using the Sobel edge detector
    edges = edge(img, 'sobel');
    
    % Calculate edge density
    edgeDensity = sum(edges(:)) / numel(edges);
    
    % Calculate edge orientation histogram
    [Gx, Gy] = imgradientxy(img);
    [~, Gdir] = imgradient(Gx, Gy);
    edgeOrientationHist = histcounts(Gdir(edges), -180:20:180);
    
    % Combine edge features
    edgeFeatures = [edgeDensity, edgeOrientationHist];
end

%% Function to extract Gabor features from an image
function gaborFeatures = extractGaborFeatures(img)
    wavelength = 4; % Example value, adjust as needed
    orientation = 0; % Example value, adjust as needed
    gaborArray = gabor(wavelength, orientation);
    gaborMag = imgaborfilt(img, gaborArray);
    gaborFeatures = mean(gaborMag(:));
end

%% Function to extract Color Histogram features from an image
function colorHist = extractColorHistogram(img)
    % Check if the image is grayscale
    if size(img, 3) == 1
        % Convert grayscale image to RGB
        img = cat(3, img, img, img);
    end
    
    % Convert image to HSV color space
    hsvImg = rgb2hsv(img);
    
    % Calculate histograms for each channel
    hHist = imhist(hsvImg(:,:,1));
    sHist = imhist(hsvImg(:,:,2));
    vHist = imhist(hsvImg(:,:,3));
    
    % Combine histograms
    colorHist = [hHist(:)', sHist(:)', vHist(:)'];
end