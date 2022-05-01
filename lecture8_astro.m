%% let's make a filter to highlight point of light

% M represents our image
M = "C:\Users\kylev\Desktop\Sem6\CS3907 Comp Photo\CompPhotoFinalProject\lights.jpg";
Morg = im2double(imread(M));
M = im2double(imread(M));


%Rotate if necessary
M = imrotate(M,270);
Morg = imrotate(Morg,270);

im = double(rgb2gray(M))./255;
F = fspecial('Gaussian',[21 21],1.3) - fspecial('Gaussian',[21 21],7);

%% list of possible points
M2 = [];
M2 = M .* 0;
for mx = 1:size(M,3);
    im2 = imfilter(M(:,:,mx),F);
    M2(:,:,mx) = im2;
    imagesc(im2);
    axis off; title(mx); drawnow;
end

imagesc(max(M2,[],3));

%% oy, the boundaries look bright.  let's squash that.
M2(1:11,:,:) = 0;
M2(end-11:end,:,:) = 0;
M2(:,1:11,:) = 0;
M2(:,end-11:end,:) = 0;

% Let's find some features:


%% USE THIS TO CHANGE THE LENGTH OF OUR LINES
lineLength = 2;




for threshold = 1:3
    numPoints = 0;
    Q = [];
    for mx = 1:size(M2,3);
        starImage = M2(:,:,mx) > (threshold*.05) + .36;
        R = regionprops(starImage);
        for rx = 1:size(R,1);
            P = R(rx).Centroid;
            P(3) = mx;                              % put in the frame number
            numPoints = numPoints + 1;
            Q(numPoints,:) = P;
        end
        disp(mx);
    end
    
%     plot3(Q(:,1),Q(:,2),Q(:,3),'.')
%     zlabel('time');
    
    
    % woah, looks like we got the "x,y" vs. "row, column" correct!
    %now that we have our points, we can apply the astigmatism effect
    imagesc(M)
    hold on;
    %we will go through our list of points Q
    for point = 1:length(Q)

        % LOWER HALF
        Xpos = floor(Q(point,1));
        Ypos = floor(Q(point,2));
        color = [M(Ypos,Xpos,1),      M(Ypos,Xpos,2),    M(Ypos,Xpos,3)];
    % color = [1,0,0];
        X = [floor(Q(point,1)),floor(Q(point,1)-(14*threshold/lineLength))];
        Y = [floor(Q(point,2)),floor(Q(point,2)+(50*threshold/lineLength))];
        M = insertShape(M,'Line',[X(1) Y(1) X(2) Y(2)],'LineWidth',4,'Color',color);
        
        X = [floor(Q(point,1)),floor(Q(point,1)-(2*14*threshold/lineLength))];
        Y = [floor(Q(point,2)),floor(Q(point,2)+(2*50*threshold/lineLength))];
        M = insertShape(M,'Line',[X(1) Y(1) X(2) Y(2)],'LineWidth',2,'Color',color);
    
        X = [min(floor(Q(point,1)),size(Morg,2)),max(floor(Q(point,1)-(3*14*threshold/lineLength)),1)];
        Y = [floor(Q(point,2)),floor(Q(point,2)+(3*50*threshold/lineLength))];
        M = insertShape(M,'Line',[X(1) Y(1) X(2) Y(2)],'LineWidth',1,'Color',color);
        
        M = M(1:size(Morg,1), 1:size(Morg,2), 1:size(Morg,3));
        % min point is middle
        
  
        %reset x y range
        X = [min(floor(Q(point,1)+(3*14*threshold/lineLength)),size(Morg,2)),max(floor(Q(point,1)-(3*14*threshold/lineLength)),1)];
        Y = [floor(Q(point,2)-(3*50*threshold/lineLength)),floor(Q(point,2)+(3*50*threshold/lineLength))];


        upper = 0;
        lower = -2;
        decreasing = logspace(upper, lower, (round(X(1)-X(2))/2)-1);
        increasing = logspace(lower, upper, (round(X(1)-X(2))/2)-1);

        % upper half blurred
%         decreasing = [decreasing 0];
%         increasing = 1-decreasing;

        % lower half blurred
        increasing = [0 increasing];
        decreasing = 1-increasing;

        %increasing = [0 increasing];

%         decreasing = logspace(upper, lower, round(X(1)-X(2))/2);
%         increasing = logspace(lower, upper, round(X(1)-X(2))/2);

        if mod((X(1) - X(2)),2) == 0
            line2 = [decreasing 0 increasing];
            line1 = [increasing 1 decreasing];
        else
            line2 = [decreasing 0 0 increasing];
            line1 = [increasing 1 1 decreasing];
        end

%         decreasing = linspace(1, 0, round(X(1)-X(2))/3);
%         increasing = linspace(0, 1, round(X(1)-X(2))/3);
%         line2 = [decreasing zeros(1,round(X(1)-X(2))/3) 0 increasing];
%         line1 = [increasing ones(1, round(X(1)-X(2))/3) 1 decreasing];


        % M is the image with lines
        % Morg is the original image
        M(max(Y(1),1):min(Y(2), size(Morg, 1)), X(2):X(1), :) = line1.* M(max(Y(1),1):min(Y(2), size(Morg, 1)),X(2):X(1), :) + line2 .* Morg(max(Y(1),1):min(Y(2), size(Morg, 1)),X(2):X(1), :);


%       Attempting over whole image
%         M(1:size(Morg, 1), 1:size(Morg,2), :) = line1.* M(1:size(Morg, 1),1:size(Morg,2), :) + line2 .* Morg(1:size(Morg, 1),1:size(Morg,2), :);
        Morg = M;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% % UPPER HALF
        Xpos = floor(Q(point,1));
        Ypos = floor(Q(point,2));
        color = [M(Ypos,Xpos,1),      M(Ypos,Xpos,2),    M(Ypos,Xpos,3)];
    % color = [1,0,0];
        X = [floor(Q(point,1)+(14*threshold/lineLength)),floor(Q(point,1))];
        Y = [floor(Q(point,2)-(50*threshold/lineLength)),floor(Q(point,2))];
        M = insertShape(M,'Line',[X(1) Y(1) X(2) Y(2)],'LineWidth',4,'Color',color);
        %plot(X,Y,'Color',color,'LineWidth',5)
        
        X = [floor(Q(point,1)+(2*14*threshold/lineLength)),floor(Q(point,1))];
        Y = [floor(Q(point,2)-(2*50*threshold/lineLength)),floor(Q(point,2))];
        M = insertShape(M,'Line',[X(1) Y(1) X(2) Y(2)],'LineWidth',2,'Color',color);
    
        X = [min(floor(Q(point,1)+(3*14*threshold/lineLength)),size(Morg,2)),max(floor(Q(point,1)),1)];
        Y = [floor(Q(point,2)-(3*50*threshold/lineLength)),floor(Q(point,2))];
        M = insertShape(M,'Line',[X(1) Y(1) X(2) Y(2)],'LineWidth',1,'Color',color);
        
        M = M(1:size(Morg,1), 1:size(Morg,2), 1:size(Morg,3));
        % min point is middle
        
  

        %reset x y range
        X = [min(floor(Q(point,1)+(3*14*threshold/lineLength)),size(Morg,2)),max(floor(Q(point,1)-(3*14*threshold/lineLength)),1)];
        Y = [floor(Q(point,2)-(3*50*threshold/lineLength)),floor(Q(point,2)+(3*50*threshold/lineLength))];

        upper = 0;
        lower = -2;
        decreasing = logspace(upper, lower, (round(X(1)-X(2))/2)-1);
        increasing = logspace(lower, upper, (round(X(1)-X(2))/2)-1);

        % upper half blurred
        decreasing = [decreasing 0];
        increasing = 1-decreasing;

        % lower half blurred
%         increasing = [0 increasing];
%         decreasing = 1-increasing;

        %increasing = [0 increasing];

%         decreasing = logspace(upper, lower, round(X(1)-X(2))/2);
%         increasing = logspace(lower, upper, round(X(1)-X(2))/2);

        if mod((X(1) - X(2)),2) == 0
            line2 = [decreasing 0 increasing];
            line1 = [increasing 1 decreasing];
        else
            line2 = [decreasing 0 0 increasing];
            line1 = [increasing 1 1 decreasing];
        end

%         decreasing = linspace(1, 0, round(X(1)-X(2))/3);
%         increasing = linspace(0, 1, round(X(1)-X(2))/3);
%         line2 = [decreasing zeros(1,round(X(1)-X(2))/3) 0 increasing];
%         line1 = [increasing ones(1, round(X(1)-X(2))/3) 1 decreasing];


        % M is the image with lines
        % Morg is the original image
        M(max(Y(1),1):min(Y(2), size(Morg, 1)), X(2):X(1), :) = line1.* M(max(Y(1),1):min(Y(2), size(Morg, 1)),X(2):X(1), :) + line2 .* Morg(max(Y(1),1):min(Y(2), size(Morg, 1)),X(2):X(1), :);
       
        %pause(1);
    end
end

imshow(M)
