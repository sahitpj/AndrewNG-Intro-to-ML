function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

z = sum(X.*(theta'),2);
h = 1./(1+(e.^(-z)));
k = size(theta,1);
J = (sum(-((1-y).*log(1-h)+(y).*log(h)))/m) +  (lambda/(2*m))*sum(theta(2:k,1).*theta(2:k,1));

grad(1,1) = (sum((h-y).*X(:,1))/m);
grad(2:k,1) =  ((sum((h-y).*X(:,2:k))')/m) + (theta(2:k,1)*lambda)/m;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta





% =============================================================

end
