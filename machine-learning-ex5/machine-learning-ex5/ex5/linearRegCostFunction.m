function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
k = size(theta,1);
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

J = sum(((X*theta)-y).^2)/(2*m) + (lambda/(2*m))*sum(theta(2:k,:).^2);
grad(1,1) = (sum((((X*theta)-y).*X),1)'(1,1)/m);
grad(2:k,1) = (sum((((X*theta)-y).*X),1)'(2:k,1)/m) + (lambda/m)*(theta(2:k,1));
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%












% =========================================================================

grad = grad(:);

end
