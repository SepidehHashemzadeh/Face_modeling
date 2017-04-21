%% Pattern Recognition and Machine Learning, Prof. Zhu, Project 1, Part 1
%  Name: Arash Vahabpour, UID: 004592220

%% -------- 1 --------
%% Preparation
clc
close all

pathstr = fileparts(mfilename('fullpath'));

%% Reading files

% note: there are 150 images in the training set, and 27 in the test set,
% i.e. a total of 177 images.

face_size = [256 256];
P = zeros(prod(face_size), 177);
index = 0;
for i = 0:177
    if i == 103
        continue %% the 103rd picture does not exist!
    end
    index = index + 1;
    P(:, index) = 1/256 * double(reshape(imread(strcat(pathstr,'/face_data/face/face',num2str(i,'%03.0f'),'.bmp')), prod(face_size),1));
end


%% Dimensionality reduction

mean_face = mean(P(:,1:150) ,2); % averaging in the second dimension (i.e. different images)
P_no_DC = P - repmat(mean_face, 1, 177);
covariance = P_no_DC(:,1:150)' * P_no_DC(:,1:150);
[V,D] = eig(covariance);

D = diag(D); %Flattening the eigenvalues matrix into a vector

[~,I] = sort(D, 'descend');
V = V(:,I); %sorting eigen vectors
V = P_no_DC(:,1:150) * V; %retreiving original eigen vectors

%% Ploting the results
vec_imshow = @(I, mean_face, face_size) imshow(reshape(I + mean_face,face_size));
% 
% figure
plot(D(I))
%  
%  figure
vec_imshow(zeros(256*256, 1), mean_face, face_size)
% % % title 'mean face'
 
% figure
for i = 1:20
      subplot(4, 5, i);
      vec_imshow(V(:,i), mean_face, face_size)
end

%% Normalizing eigen vectors

for i=1:size(V,2)
    V(:,i) = V(:,i) / norm(V(:,i),2);
end

%% Calculating the reconstruction error

dimensionality = 1:5:150;
reconstruction_error = zeros(27,length(dimensionality));

for K = dimensionality % Target dimensionality
    descriptors = V(:,1:K)' * P_no_DC(:,151:177);
    face_test_reconstructed = V(:,1:K) * descriptors;
    for i = 1:27
        reconstruction_error(i, dimensionality == K) = ...
            mean((P_no_DC(:,150+i) - face_test_reconstructed(:,i)) .^ 2); % for each image and each K
    end    
end

reconstruction_error = mean(reconstruction_error) * 256^2; % average over test images + take to the 256 level scal

figure
plot(dimensionality, reconstruction_error);
title 'reconstruction error'

%% -------- 2 --------
%% Reading files

% note: there are 150 images in the training set, and 27 in the test set,
% i.e. a total of 177 images.

landmark_size = [87 2];
L = zeros(prod(landmark_size), 177);
index = 0;
for i = 0:177
    if i == 103
        continue %% the 103rd picture does not exist!
    end
    index = index + 1;
    fileID = fopen(strcat(pathstr,'/face_data/landmark_87/face',num2str(i,'%03.0f'),'_87pt.dat'));
    formatspec = '%f';
    A = fscanf(fileID, formatspec);
    L(:, index) = A(2:end);
    fclose(fileID);
end

%% Dimensionality reduction

mean_warping = mean(L(:,1:150) ,2); % averaging in the second dimension (i.e. different images)
L_no_DC = L - repmat(mean_warping, 1, 177);
covariance = L_no_DC(:,1:150)' * L_no_DC(:,1:150);
[V_warping,D_warping] = eig(covariance);

D_warping = diag(D_warping); %Flattening the eigenvalues matrix into a vector

[~,I_warping] = sort(D_warping, 'descend');
V_warping = V_warping(:,I_warping); %sorting eigen vectors
V_warping = L_no_DC(:,1:150) * V_warping; %retreiving original eigen vectors

%% Ploting the results

% figure
% show_landmarks(mean_warping, face_size, landmark_size);
% title 'mean warping'
% 
% figure
% for i = 1:5
%     subplot(5, 1, i);
%     show_landmarks(mean_warping + V_warping(:,i), face_size, landmark_size);
%     axis 'equal'
% end

%% Normalizing eigen vectors

for i=1:size(V_warping,2)
    V_warping(:,i) = V_warping(:,i) / norm(V_warping(:,i),2);
end

%% Calculating the reconstruction error

dimensionality = 1:5:150;
reconstruction_error = zeros(27,length(dimensionality));

for K = dimensionality % Target dimensionality
    descriptors = V_warping(:,1:K)' * L_no_DC(:,151:177);
    landmark_test_reconstructed = V_warping(:,1:K) * descriptors;
    for i = 1:27
        reconstruction_error(i, dimensionality == K) = ...
            mean((L_no_DC(:,150+i) - landmark_test_reconstructed(:,i)) .^ 2); % for each image and each K
    end    
end

reconstruction_error = mean(reconstruction_error) * 256^2; % average over test images + take to the 256 level scal

figure
plot(dimensionality, reconstruction_error);
title 'reconstruction error'

%% -------- 3 --------
%% (i)
K = 10;
descriptors = V_warping(:,1:K)' * L_no_DC;
landmark_test_reconstructed = V_warping(:,1:K) * descriptors;

%% (ii)
P_test_warped = uint8(zeros([face_size 177]));

for i = 1:177
%     hold on
%     A= reshape(mean_warping + landmark_test_reconstructed(:,i), landmark_size(2), landmark_size(1))';
%     plot(A(:,1),A(:,2), '.')
%     hold off
    
    P_test_warped(:,:, i) = warpImage_kent(uint8(256 *reshape(P(:, i),face_size)), ...
        (reshape(mean_warping + landmark_test_reconstructed(:,i), landmark_size(2), landmark_size(1)))', ... %reshaping to a 87x2 matrix
        (reshape(mean_warping, landmark_size(2), landmark_size(1)))');
    if i > 173
        figure
        imshow(P_test_warped(:,:,i));
    end
end


%% (iii)
%% Dimensionality reduction

if isa(P_test_warped, 'uint8')
    P_test_warped = 1/256 * double(reshape(P_test_warped, prod(face_size), 177));
end

mean_warped_face = mean(P_test_warped(:,1:150) ,2); % averaging in the second dimension (i.e. different images)
P_test_warped_no_DC = P_test_warped - repmat(mean_warped_face, 1, 177);
covariance = P_test_warped_no_DC(:,1:150)' * P_test_warped_no_DC(:,1:150);
[V_test_warped,D_test_warped] = eig(covariance);

D_test_warped = diag(D_test_warped); %Flattening the eigenvalues matrix into a vector

[~,I_test_warped] = sort(D_test_warped, 'descend');
V_test_warped = V_test_warped(:,I_test_warped); %sorting eigen vectors
V_test_warped = P_test_warped_no_DC(:,1:150) * V_test_warped; %retreiving original eigen vectors

%% Ploting the results
% vec_imshow = @(I, face_size) imshow(reshape(I + mean_face,face_size));
% 
% figure
% plot(D(I))
% 
% figure
% vec_imshow(zeros(256*256, 1), face_size)
% title 'mean face'
% 
% figure
% for i = 1:20
%     subplot(4, 5, i);
%     vec_imshow(V(:,i), face_size)
% end

%% Normalizing eigen vectors

for i=1:size(V_test_warped,2)
    V_test_warped(:,i) = V_test_warped(:,i) / norm(V_test_warped(:,i),2);
end

%% Calculating the reconstruction error

dimensionality = 1:5:150;
reconstruction_error = zeros(27,length(dimensionality));

for K = dimensionality % Target dimensionality
    descriptors = V_test_warped(:,1:K)' * P_test_warped_no_DC(:,151:177);
    face_test_reconstructed = V_test_warped(:,1:K) * descriptors;
    face_test_reconstructed_unwarped = uint8(zeros([face_size 27]));
    
    for i = 1:27
        face_test_reconstructed_unwarped(:,:, i) = ...
            warpImage_kent(uint8(256 *reshape(face_test_reconstructed(:, i) + mean_warped_face,face_size)), ...
            (reshape(mean_warping, landmark_size(2), landmark_size(1)))', ...
            (reshape(mean_warping + landmark_test_reconstructed(:,150 + i), landmark_size(2), landmark_size(1)))');

        reconstruction_error(i, dimensionality == K) = ...
            mean((P(:,150+i) - 1/256 * double(reshape(face_test_reconstructed_unwarped(:,:,i), prod(face_size), 1))) .^ 2); % for each image and each K
    end    
end

reconstruction_error = mean(reconstruction_error) * 256^2; % average over test images + take to the 256 level scal

figure
plot(dimensionality, reconstruction_error);
title 'reconstruction error'

%% Synthesizing Faces

number_of_synth_faces = 20;
dimensionality = 10;

face_weights = rand(dimensionality,number_of_synth_faces);
face_weights = (face_weights ./ repmat(sum(face_weights), dimensionality , 1)) .* repmat(sqrt(D_test_warped(I_test_warped(1:dimensionality))), 1, number_of_synth_faces); % scaling for proper weights

landmark_weights = rand(dimensionality,number_of_synth_faces);
landmark_weights = landmark_weights ./ repmat(sum(landmark_weights), dimensionality , 1) .* repmat(sqrt(D_warping(I_warping(1:dimensionality))), 1, number_of_synth_faces);

face_synth = V_test_warped(:,1:dimensionality) * face_weights;
landmark_synth = V_warping(:,1:dimensionality) * landmark_weights;

figure
for i = 1:number_of_synth_faces
    subplot(4,5,i)
    imshow(...
        warpImage_kent(uint8(256 *reshape(face_synth(:, i) + mean_warped_face,face_size)), ...
        (reshape(mean_warping, landmark_size(2), landmark_size(1)))', ...
        (reshape(mean_warping + landmark_synth(:,i), landmark_size(2), landmark_size(1)))'));
end