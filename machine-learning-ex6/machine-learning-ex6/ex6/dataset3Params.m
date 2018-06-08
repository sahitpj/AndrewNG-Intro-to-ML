function [c1, s1] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
CList = [0.01;0.03;0.1;0.3;1;3;10;30;100;300]
SigmaList = [0.01;0.03;0.1;0.3;1;3;10;30;100;300]
l = zeros(64,1)
k = 1
for i = 1:8
  C = CList(i,1)
  for j = 1:8
    sigma = SigmaList(j,1)
    model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma))
    predictions = svmPredict(model, Xval) 
    l(k,1) = mean(double(predictions ~= yval))
    if k == 1
      t = l(1,1)
      c1 = C
      s1 = sigma
    else
      if l(k,1) < t
        t = l(k,1)
        c1 = C
        s1 = sigma
      endif
    endif
    k = k + 1
  endfor
disp(c1)
disp(s1)
  

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%







% =========================================================================

end
