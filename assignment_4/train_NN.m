function [model] = train_NN(X_train, y)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

size_train = size(X_train);
N_inputs = size_train(1);

%DEFINE THE NETWORK ARCHITECTURE
numInputs=1;  %NOTE: THIS DOES NOT MEAN THAT THERE IS ONLY ONE INPUT 
              %      DIMENSION TO THE NET! THE DIMENSIONS IS ASSIGNED
              %      IS ASSIGNED BELOW AFTER NET INITIALIZATION
numLayers=2;
biasConnect=[1;1];       %   - numLayers-by-1 Boolean vector, zeros.

inputConnect=[1;
              0];% - numLayers-by-numInputs Boolean matrix, zeros.

layerConnect=[0 0;
              1 0];   %  - numLayers-by-numLayers Boolean matrix, zeros.

outputConnect=[0 1];  % - 1-by-numLayers Boolean vector, zeros.

net=   network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect);
no_of_hidden_nodes=100;  %THIS IS THE NUMBER OF HIDDEN NODES IN THE NET TO BE TRAINED
net.layers{1}.dimensions = no_of_hidden_nodes;
net.layers{2}.dimensions = 1;
net.inputs{1}.size = N_inputs;% dimensions of input vectors




%START WEIGHTS
std_start_weights=10^(-2);

% Good guesses

% Guess for 8 first a
alpha=10;

% Guess for 8 first b
a=2.5;
b=3.5;
c=1.5;
d=2.5;
e=4.5;
f=5;
g=1.5;
h=0.5;

% Guess for 8 first w
w1 = [1 0];
w2 = [1 0];
w3 = [0 1];
w4 = [0 1];
w5 = [1 1];
w6 = [1 1];
w7 = [-1 1];
w8 = [-1 1];

b_guess=std_start_weights*randn(no_of_hidden_nodes,1);
b_guess(1:8) = [-a -b -c -d -e -f -g -h];

W_guess = std_start_weights*randn(no_of_hidden_nodes,N_inputs);
W_guess(1:8, :) = [w1; w2; w3; w4; w5; w6; w7; w8];

a_guess=std_start_weights*randn(1,no_of_hidden_nodes)';
a = [alpha, -alpha, alpha, -alpha, alpha, -alpha, alpha, -alpha]';
a_guess(1:8) = a;

yo_guess=std_start_weights*randn;

net.IW{1,1}=W_guess;
net.b{1}=b_guess;
net.LW{2,1}=a_guess';
net.b{2}=yo_guess';


%MAKE HIDDEN LAYER NON-LINEAR (DEFAULT IS 'purelin')
net.layers{1}.transferFcn='tansig';


%TRAIN THE NETWORK

%Training function
net.trainFcn = 'traingdx'  %Gradient descent with momentum and adaptive steps
%net.trainFcn ='traingda'; %Gradient descent with adaptvive steps
%net.trainFcn ='trainlm';  %Levenberg-Marquard
%net.trainParam.mu=10^(-6); %This is only for 'trainlm'

% Set up Division of Data for Training, Validation, Testing
% NOTE: Here we do not use any built-in validation or test
net.divideFcn = 'dividerand';
net.divideParam.trainRatio = 100/100;
net.divideParam.valRatio = 0/100;
net.divideParam.testRatio = 0/100;


% Set training parameters
initial_learning_rate=10^(-8);
max_no_of_epochs=100000;
%performance_goal=10^(-5);
performance_goal=0.0013;
%performance_goal=0.055;
minimum_gradient_threshold=10^(-10); 

net.trainParam.lr=initial_learning_rate;
net.trainParam.epochs=max_no_of_epochs;
net.trainParam.goal=performance_goal;
net.trainParam.min_grad=minimum_gradient_threshold;



% Call train which performs the actual training
[trained_net,tr] = train(net,X_train,y);

model = trained_net;


% %PRESENT PLOTS
% figure
% subplot(211)
% semilogy(tr.perf,'r')
% title('Performance')
% xlabel('No of iterations')
% grid
% subplot(212)
% semilogy(tr.gradient,'r')
% title('Gradient')
% xlabel('No of iterations')
% grid
%  
% y_hat_train = trained_net(X_train);
% % 
% rmse_train=sqrt(mean((y'-y_hat_train).^2))
% % 
% figure, plot(y_hat_train,y,'.')
% grid
% title('TRAINING DATA: yhat and y')
% title(['TRAINING DATA: yhat and y        RMSEtrain=' num2str(rmse_train)])
% % 
% % 
% % %WRITE OUT WEIGHTS OBTIANED AFTER TRAINING
% W_hat=trained_net.IW{1};
% b_hat=trained_net.b{1};
% a_hat=trained_net.LW{2,1}';
% yo_hat=trained_net.b{2};
% % 
% % 
% % 
% % 
% % %WRITE OUT TRUE, GUESS AND RESULTING NETWORK PARAMETERS 
% 
% W_guess
% 
% W_hat
% 
% 
% figure
% subplot(211)
% plot(W_hat','.-')
% grid
% title('WEIGHTS FOR ALL HIDDEN NEURONS - VARIABLE IMPORTANCE: Rows of W_{hat}')
% xlabel('input dimensions 1-10')
% subplot(212)
% plot(W_hat,'.-')
% title('WEIGHTS FOR ALL THE HIDDEN NODES - LOADINGS (NODE INFLUENCES): Columns of  W_{hat}')
% grid
% xlabel('index of hidden node')
% 
% b_hat
% 
% a_hat
% 
% 
% 
% yo_hat

end

