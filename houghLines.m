% This code was written by Arjun Devarajan

function numLines = houghLines(image)
    % Create a grayscaled version of the image
    grayscaleImage = rgb2gray(image);
    
    % Run the canny edge detection method on the grayscaled image
    cannyEdge = edge(grayscaleImage,'canny');
    
    % Run the Hough transform on the canny edge detection results
    [houghResults,theta,rho] = hough(cannyEdge);
    
    % Find the peaks of the Hough transform
    houghPeakResults = houghpeaks(houghResults,5,'threshold',ceil(0.3*max(houghResults(:))));

    % Find the lines using the peaks of the Hough transform
    lines = houghlines(cannyEdge,theta,rho,houghPeakResults,'FillGap',5,'MinLength',7);

    % Find the total number of lines in the image and return this
    numLines = length(lines);
end