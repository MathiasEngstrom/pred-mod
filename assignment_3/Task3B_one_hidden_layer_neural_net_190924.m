

close all force 
clear all

%GENERATE TRAINING DATA USING A ONE HIDDEN LEYER NETWORK
N_train=10000;
N_inputs=10;
N_hidden=4;  %THIS IS THE NO OF HIDDEN NODES IN THE DATA GENERATING NET
sigma_noise=0;
X_train=randn(N_inputs,N_train);

%CHOOSE WEIGHTS RANDOMLY
W=randn(N_hidden,N_inputs);  %Weights to hidden layer from inputs
b=randn(N_hidden,1);         %BIases in hidden layer
a=randn(N_hidden,1);         %Weights from hidden layer to output
yo=randn;                    %Bias of output node

%CHOOSE WEIGHTS MANUALLY
W=[2 -1;3 2;4 -3;5 -2];
b=[1 2 3 4]';
a=[1 2 3 4]';
yo=1;

%Add extra zero weights
W=[W zeros(4,N_inputs-2)];



B=repmat(b,1,N_train);
H=tanh(W*X_train+B);
y=H'*a+yo;
y=y+randn(size(y))*sigma_noise;




%DEFINE THE NETWORK ARCHITECTURE
numInputs=1;  %NOTE: THIS DOES NOT MEAN THAT THERE IS ONLY ONE INPUT 
              %      DIMENSION TO THE NET! THE DIMENSIONS IS ASSIGNED
              %      IS ASSIGNED BELOW AFTER NET INITIALIZATION
numLayers=2;
biasConnect=[1;1];       %   - numLayers-by-1 Boolean vector, zeros.
inputConnect=[1;
              0];     % - numLayers-by-numInputs Boolean matrix, zeros.
layerConnect=[0 0;
              1 0];   %  - numLayers-by-numLayers Boolean matrix, zeros.
outputConnect=[0 1];  % - 1-by-numLayers Boolean vector, zeros.
net=   network(numInputs,numLayers,biasConnect,inputConnect,layerConnect,outputConnect);
no_of_hidden_nodes=4;  %THIS IS THE NUMBER OF HIDDEN NODES IN THE NET TO BE TRAINED
net.layers{1}.dimensions = no_of_hidden_nodes;
net.layers{2}.dimensions = 1;
net.inputs{1}.size = N_inputs;% dimensions of input vectors




%START WEIGHTS
std_start_weights=10^(-2);
W_guess=std_start_weights*randn(no_of_hidden_nodes,N_inputs);
b_guess=std_start_weights*randn(no_of_hidden_nodes,1);
a_guess=std_start_weights*randn(1,no_of_hidden_nodes)';
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
max_no_of_epochs=100000
performance_goal=10^(-5)
%performance_goal=0.013;
%performance_goal=0.055;
minimum_gradient_threshold=10^(-10); 

net.trainParam.lr=initial_learning_rate;
net.trainParam.epochs=max_no_of_epochs;
net.trainParam.goal=performance_goal;
net.trainParam.min_grad=minimum_gradient_threshold;



% Call train which performs the actual training
[trained_net,tr] = train(net,X_train,y');




%PRESENT PLOTS
figure
subplot(211)
semilogy(tr.perf,'r')
title('Performance')
xlabel('No of iterations')
grid
subplot(212)
semilogy(tr.gradient,'r')
title('Gradient')
xlabel('No of iterations')
grid

y_hat_train = trained_net(X_train);

rmse_train=sqrt(mean((y'-y_hat_train).^2))

figure, plot(y_hat_train,y,'.')
grid
title('TRAINING DATA: yhat and y')
title(['TRAINING DATA: yhat and y        RMSEtrain=' num2str(rmse_train)])


%WRITE OUT WEIGHTS OBTIANED AFTER TRAINING
W_hat=trained_net.IW{1};
b_hat=trained_net.b{1};
a_hat=trained_net.LW{2,1}';
yo_hat=trained_net.b{2};


%CREATE LARGE TEST DATA SET
N_test=10000;
X_test=randn(N_inputs,N_test);
B_test=repmat(b,1,N_test);
H_test=tanh(W*X_test+B_test);
y_test=H_test'*a+yo;
epsilon_test=randn(size(y_test))*sigma_noise;
y_test=y_test+epsilon_test;
smallest_possible_rmse_test=sqrt(mean(epsilon_test.^2))

%PERFORMANCE ON TEST DATA
y_hat_test = trained_net(X_test);
rmse_test=sqrt(mean((y_test'-y_hat_test).^2))
figure, plot(y_hat_test,y_test,'.')
grid
title(['EXTERNAL TEST DATA: yhattest and ytest        RMSEtest=' num2str(rmse_test)])



%WRITE OUT TRUE, GUESS AND RESULTING NETWORK PARAMETERS 
W
W_guess

W_hat


figure
subplot(211)
plot(W_hat','.-')
grid
title('WEIGHTS FOR ALL HIDDEN NEURONS - VARIABLE IMPORTANCE: Rows of W_{hat}')
xlabel('input dimensions 1-10')
subplot(212)
plot(W_hat,'.-')
title('WEIGHTS FOR ALL THE HIDDEN NODES - LOADINGS (NODE INFLUENCES): Columns of  W_{hat}')
grid
xlabel('index of hidden node')

b_hat

a_hat



yo_hat




