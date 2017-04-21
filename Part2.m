%% Pattern Recognition and Machine Learning, Prof. Zhu, Project 1, Part 2
%  Name: Arash Vahabpour, UID: 004592220

%% Preparation and reading images
clc
clear
close all

female_training_size = 75;
female_test_size = 10;
male_training_size = 78;
male_test_size = 10;

face_size = [256 256];
female_training_images = zeros([prod(face_size) female_training_size]);
female_test_images = zeros([prod(face_size) female_test_size]);
male_training_images = zeros([prod(face_size) male_training_size]);
male_test_images = zeros([prod(face_size) male_test_size]);

pathstr = fileparts(mfilename('fullpath'));

index = 0;
for i = 0:female_training_size - 1
    index = index + 1;
    female_training_images(:, index) = double(reshape(imread(strcat... %scale to [0,1] interval
                  (pathstr,'\\face_data\\female_face\\face',num2str(i,'%03.0f'),'.bmp')), prod(face_size),1));
end
female_mean_face = mean(female_training_images , 2);

index = 0;
for i = female_training_size:female_training_size + female_test_size - 1
    index = index + 1;
    female_test_images(:, index) = double(reshape(imread(strcat... %scale to [0,1] interval
                  (pathstr,'\\face_data\\female_face\\face',num2str(i,'%03.0f'),'.bmp')), prod(face_size),1));
end

index = 0;
for i = 0:male_training_size
    if i == 57
        continue %% the 57th picture does not exist!
    end
    index = index + 1;
    male_training_images(:, index) = double(reshape(imread(strcat... %scale to [0,1] interval
                  (pathstr,'\\face_data\\male_face\\face',num2str(i,'%03.0f'),'.bmp')), prod(face_size),1));
end
male_mean_face = mean(male_training_images , 2);

index = 0;
for i = male_training_size + 1:male_training_size + male_test_size
    index = index + 1;
   male_test_images(:, index) = double(reshape(imread(strcat... %scale to [0,1] interval
                  (pathstr,'\\face_data\\male_face\\face',num2str(i,'%03.0f'),'.bmp')), prod(face_size),1));
end
 
%% -------- 5 --------
%%

N = 256;
C = zeros(N^2, female_training_size + male_training_size);
C(:, 1:female_training_size) = female_training_images - repmat(male_mean_face, 1, female_training_size);
C(:, female_training_size + 1:end) = male_training_images - repmat(female_mean_face, 1, male_training_size);
B = C' * C;

[V,D] = eig(B);
D = diag(D); %Flattening the eigenvalues matrix into a vector


%% Normalizing eigen vectors

for i=1:size(V,2)
    V(:,i) = V(:,i) / norm(V(:,i),2);
end

%%
A = zeros(N^2, female_training_size + male_training_size);
for i=1:female_training_size + male_training_size
    A(:,i) = (sqrt(D(i)) / norm(C*V(:,i) ,2) ) * C*V(:,i);
end

y = A' * (female_mean_face - male_mean_face);
z = (diag(D)^2 * V') \ y;
w = real(C * z);

% displaying the result
vec_imshow = @(I, mean_face, face_size) imshow(reshape(I + mean_face,face_size));
vec_imshow(uint8((128 * w / max(abs(w)) + 127)), ... % rescale
    uint8(zeros(prod(face_size),1)), face_size);

female_test_output = w' * female_test_images;
male_test_output = w' * male_test_images;
female_training_output = w' * female_training_images;
male_training_output = w' * male_training_images;

figure;
hold on;
scatter(female_test_output , ones(10,1) , '*', 'b');
scatter(male_test_output, ones(10,1) ,  '*' ,'r');
hold off;
ylim([0,3]);

%% Fitting Gaussian curves to the data
[muhat_f , sigmahat_f] = normfit(female_training_output);
[muhat_m , sigmahat_m] = normfit(male_training_output);

%% -------- 6 --------
%% Reading files

landmark_size = [87 2];
female_training_landmarks = zeros(prod(landmark_size),female_training_size);
female_test_landmarks = zeros(prod(landmark_size),female_test_size);
male_training_landmarks = zeros(prod(landmark_size),female_training_size);
male_test_landmarks = zeros(prod(landmark_size),female_test_size);

index = 0;
for i = 0:female_training_size - 1
    index = index + 1;
    fileID = fopen(strcat(pathstr,'\\face_data\\female_landmark_87\\face',num2str(i,'%03.0f'),'_87pt.txt'));
    formatspec = '%f';
    A = fscanf(fileID, formatspec);
    female_training_landmarks(:, index) = A(:);
    fclose(fileID);
end

index = 0;
for i = female_training_size:female_training_size + female_test_size - 1
    index = index + 1;
    fileID = fopen(strcat(pathstr,'\\face_data\\female_landmark_87\\face',num2str(i,'%03.0f'),'_87pt.txt'));
    formatspec = '%f';
    A = fscanf(fileID, formatspec);
    female_test_landmarks(:, index) = A(:);
    fclose(fileID);
end


index = 0;
for i = 0:male_training_size
    if i == 57
        continue % 57th file does not exist
    end
    index = index + 1;
    fileID = fopen(strcat(pathstr,'\\face_data\\male_landmark_87\\face',num2str(i,'%03.0f'),'_87pt.txt'));
    formatspec = '%f';
    A = fscanf(fileID, formatspec);
    male_training_landmarks(:, index) = A(:);
    fclose(fileID);
end

index = 0;
for i = male_training_size + 1:male_training_size + male_test_size 
    index = index + 1;
    fileID = fopen(strcat(pathstr,'\\face_data\\male_landmark_87\\face',num2str(i,'%03.0f'),'_87pt.txt'));
    formatspec = '%f';
    A = fscanf(fileID, formatspec);
    male_test_landmarks(:, index) = A(:);
    fclose(fileID);
end

%% Warping

mean_warping = mean([female_training_landmarks , male_training_landmarks] , 2 );
female_mean_landmark = mean(female_training_landmarks , 2);
male_mean_landmark = mean(male_training_landmarks , 2);

female_warped_training_images = zeros(size(female_training_images));
for i=1:female_training_size
    warped_face = warpImage_kent(uint8(reshape(female_training_images(:,i) , face_size)),...
            (reshape(female_training_landmarks(:,i), landmark_size(2), landmark_size(1)))', ... %reshaping to a 87x2 matrix
            (reshape(mean_warping, landmark_size(2), landmark_size(1)))');
    female_warped_training_images(:,i) = reshape(warped_face, prod(face_size), 1);
end
female_mean_warped_face = mean(female_warped_training_images , 2);


female_warped_test_images = zeros(size(female_test_images));
for i=1:female_test_size
    warped_face = warpImage_kent(uint8(reshape(female_test_images(:,i) , face_size)),...
            (reshape(female_test_landmarks(:,i), landmark_size(2), landmark_size(1)))', ... %reshaping to a 87x2 matrix
            (reshape(mean_warping, landmark_size(2), landmark_size(1)))');
    female_warped_test_images(:,i) = reshape(warped_face, prod(face_size), 1);
end

male_warped_training_images = zeros(size(male_training_images));
for i=1:male_training_size
    warped_face = warpImage_kent(uint8(reshape(male_training_images(:,i) , face_size)),...
            (reshape(male_training_landmarks(:,i), landmark_size(2), landmark_size(1)))', ... %reshaping to a 87x2 matrix
            (reshape(mean_warping, landmark_size(2), landmark_size(1)))');
    male_warped_training_images(:,i) = reshape(warped_face, prod(face_size), 1);
end
male_mean_warped_face = mean(male_warped_training_images , 2);


male_warped_test_images = zeros(size(male_test_images));
for i=1:male_test_size
    warped_face = warpImage_kent(uint8(reshape(male_test_images(:,i) , face_size)),...
            (reshape(male_test_landmarks(:,i), landmark_size(2), landmark_size(1)))', ... %reshaping to a 87x2 matrix
            (reshape(mean_warping, landmark_size(2), landmark_size(1)))');
    male_warped_test_images(:,i) = reshape(warped_face, prod(face_size), 1);
end

%% Fisher Face

C_landmarks = [female_training_landmarks male_training_landmarks] ...
    - [repmat(female_mean_landmark, 1, female_training_size) ...
    repmat(male_mean_landmark, 1, male_training_size)];

S_landmarks = C_landmarks * C_landmarks';
w_landmarks = S_landmarks \ (female_mean_landmark - male_mean_landmark);
w_landmarks = w_landmarks / norm(w_landmarks,2);




% warped images fisherface

C_landmarks = [female_training_images male_training_images] ...
    - [repmat(female_mean_face, 1, female_training_size) ...
    repmat(male_mean_face, 1, male_training_size)];

B = C' * C;

[V,D] = eig(B);
D = diag(D);

for i=1:size(V,2)
    V(:,i) = V(:,i) / norm(V(:,i),2);
end

A = zeros(N^2 , female_training_size + male_training_size);
for i=1:female_training_size + male_training_size
    A(:,i) = (sqrt(D(i)) / norm(C*V(:,i) ,2) ) * C*V(:,i);
end

y = A' * (female_mean_warped_face - male_mean_warped_face);

z = (diag(D)^2 * V') \ y;
w_warped = real(C * z);

vec_imshow(uint8((128 * w_warped / max(abs(w_warped)) + 127)), ... % rescale
    uint8(zeros(prod(face_size),1)), face_size);

female_warped_test_output = [w_landmarks' * female_test_landmarks, w_warped' * female_warped_test_images];
male_warped_test_output = [w_landmarks' * male_test_landmarks, w_warped' * male_warped_test_images];
female_warped_training_output = [w_landmarks' * female_training_landmarks, w_warped' * female_warped_training_images];
male_warped_training_output = [w_landmarks' * male_training_landmarks, w_warped' * male_warped_training_images];
 

%% Illustrating the results
figure;
hold on;
scatter( female_warped_test_output(:,1) , female_warped_test_output(:,2) , '*');
scatter( male_warped_test_output(:,1) , male_warped_test_output(:,2) , '.' , 'r');
hold off;

figure;
hold on
scatter( female_warped_training_output(:,1) , female_warped_training_output(:,2) , '*');
scatter( male_warped_training_output(:,1) , male_warped_training_output(:,2) , '.' , 'r');
hold off;

