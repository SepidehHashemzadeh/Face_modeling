function show_landmarks(A, face_size, landmark_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    A = (reshape(A, landmark_size(2), landmark_size(1)))'; % assign the values to a proper matrix
    plot(A(:,1), A(:,2), '.')
    xlim([0 face_size(1)]);
    ylim([0 face_size(1)]); 
    set(gca, 'Ydir', 'reverse');
end