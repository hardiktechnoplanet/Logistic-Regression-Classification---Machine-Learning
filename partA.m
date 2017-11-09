% printing option
more off;
clc
clear all
close all

% read files
D_tr = csvread('spambasetrain.csv'); 
D_ts = csvread('spambasetest.csv');  

% construct x and y for training and testing
X_tr = D_tr(:, 1:end-1);
y_tr = D_tr(:, end);
X_ts = D_ts(:, 1:end-1);
y_ts = D_ts(:, end);

% number of training / testing samples
n_tr = size(D_tr, 1);
n_ts = size(D_ts, 1);

% add 1 as a feature
X_tr = [ones(n_tr, 1) X_tr];
X_ts = [ones(n_ts, 1) X_ts];

% perform gradient descent :: logistic regression
n_vars = size(X_tr, 2);              % number of variables
lr = 10^(-6);                           % learning rate
w = zeros(n_vars, 1);                % initialize parameter w
tolerance = 1e-2;                    % tolerance for stopping criteria


accuts = zeros(1000,1);
accutr = zeros(1000,1);

iter = 0;                            % iteration counter
max_iter = 1000;                     % maximum iteration
while true
    iter = iter + 1;                 % start iteration

    % calculate gradient
    grad = zeros(n_vars, 1);         % initialize gradient
    for j=1:n_vars
        grad(j) =((X_tr(:,j)')*y_tr - (X_tr(:,j)')*(1./(1 + exp(-1*X_tr*w))));    % compute the gradient with respect to w_j here
    end

    % take step
    w_new = w + (lr*grad);              % take a step using the learning rate
    
     printf('iter = %d, mean abs gradient = %0.3f\n', iter, mean(abs(grad)));
     fflush(stdout);

    % stopping criteria and perform update if not stopping
    if mean(abs(grad)) < tolerance
        w = w_new;
        break;
    else
        w = w_new;
    end
% use w for prediction
prediction_ts = zeros(n_ts, 1);               % initialize prediction vector
ts = 0;
for i=1:n_ts
    prediction_ts(i) = exp(X_ts(i,:)*w)/(1 + exp(X_ts(i,:)*w)) >= 0.5;               % compute your prediction
    if (prediction_ts(i) == y_ts(i))
        ts = ts + 1;
    end
end

% calculate testing accuracy
accuts(iter) = (ts/n_ts)*100;

% repeat the similar prediction procedure to get training accuracy
prediction_tr = zeros(n_tr, 1);               % initialize prediction vector
tr = 0;
for i=1:n_tr
    prediction_tr(i) = exp(X_tr(i,:)*w)/(1 + exp(X_tr(i,:)*w)) >= 0.5;               % compute your prediction
    if (prediction_tr(i) == y_tr(i))
        tr = tr + 1;
    end
end

% calculating training accuracy
accutr(iter) = (tr/n_tr)*100;
    if iter >= max_iter 
        break;
    end
end
figure
plot(1:1000,accutr,1:1000,accuts)
xlabel('number of iterations');
ylabel('accuracy in percentage');