% MMD^2 evaluation
% Using 2000 samples each (more than enough)
testnum = 7;
disp(['test ' num2str(testnum)])

tic;
if testnum == 3
    % Test1 - random noise to same random noise
    x1 = 20*rand(100, 4096);
    x2 = 20*rand(100, 4096);
elseif testnum == 2
    % Test 2 - compare gt to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(n, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i+500}';
    end
elseif testnum == 3
    % Test 3 - compare noise to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(n, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end

    x2 = 20*rand(size(x1));
    n = m;

elseif testnum == 4
    % Test 4 - compare baseline 40 to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_baseline_40/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 5
    % Test 5 - compare ours 37 to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_MCGAN128_37/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end

elseif testnum == 6
    % Test 6 - compare baseline 40 to gt in image space
    ftest = fopen('test2015files.txt');
    m = 200*10;
    x1 = zeros(m, 128*128*3);
    for i = 1:m 
        imagename = ['/scratch/kdmarino/datasets/test2015/' fgetl(ftest)];
        img = imread(imagename);
        if size(img, 3) == 1 
            img = repmat(img, [1, 1, 3]);
        end
        img = double(img) / 255;
        img = imresize(img, [128, 128]);
        x1(i, :) = img(:);
    end
    fclose(ftest);

    fbase = fopen('baseline_40_files.txt');
    n = 200*10;
    x2 = zeros(n, 128*128*3);
    for i = 1:n
        imagename = ['baseline128_40/' fgetl(fbase)];
        img = imread(imagename);
        if size(img, 3) == 1 
            img = repmat(img, [1, 1, 3]);
        end
        img = double(img) / 255;
        img = imresize(img, [128, 128]);
        x2(i, :) = img(:);
    end
    fclose(fbase);
elseif testnum == 7
    % Test 7 - compare gt to gt in image space
    ftest = fopen('test2015files.txt');
    m = 200*10;
    x1 = zeros(m, 128*128*3);
    for i = 1:m 
        imagename = ['/scratch/kdmarino/datasets/test2015/' fgetl(ftest)];
        img = imread(imagename);
        if size(img, 3) == 1 
            img = repmat(img, [1, 1, 3]);
        end
        img = double(img) / 255;
        img = imresize(img, [128, 128]);
        x1(i, :) = img(:);
    end
    fclose(ftest);

    fbase = fopen('baseline_40files.txt');
    n = 200*10;
    x2 = zeros(n, 128*128*3);
    for i = 1:n
        imagename =  ['/scratch/kdmarino/datasets/test2015/' fgetl(ftest)];
        %imagename = ['baseline128_40/' fgetl(fbase)];
        img = imread(imagename);
        if size(img, 3) == 1 
            img = repmat(img, [1, 1, 3]);
        end
        img = double(img) / 255;
        img = imresize(img, [128, 128]);
        x2(i, :) = img(:);
    end
    fclose(fbase);
elseif testnum == 8
    % Test 8 - compare gt VOC to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_VOC2007/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 9
    % Test 9 - compare baseline 37 to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_baseline_37/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 10
    % Test 10 - compare blur 21 to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_MCGAN_blur_21/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 11
    % Test 11 - compare noise 17 to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_MCGAN_noise_17/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 12
    % Test 9 again, but more samples from test
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 1000*10;
    x1 = zeros(m, 4096);
    for i = 1:1000
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_baseline_37/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 13
    % Test 10 - compare blur 34 to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_MCGAN_blur_34/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 14
    % Test 11 - compare noise 29 to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_MCGAN_noise_29/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 15
    % Test 15 - compare baseline 37 to gt COCO in conv5
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = conv5_features{i}';
    end
    load('fc7_baseline_37/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = conv5_features{i}';
    end
elseif testnum == 16
    % Test 16 - compare noise COCO to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_MCGAN_Noise_COCO/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 17
    % Test 117 - compare reg COCO to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_MCGAN_Reg_COCO/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 18
    % Test 18 - compare MCGAN VOCtrain 300 to gt VOC in fc7
    load('fc7_VOCtrain_MCGAN_300/0.mat');
    numbatch = length(fc7_features);
    m = 198*10;
    x1 = zeros(m, 4096);
    for i = 1:198
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_VOC2007/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 19
    % Test 1 - compare baseline VOCtrain 300 to gt VOC in fc7
    load('fc7_VOCtrain_baseline_300/0.mat');
    numbatch = length(fc7_features);
    m = 198*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_VOC2007/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 20
    % Test 20 - compare MCGAN VOCtrain 300 to gt VOC in fc7
    load('fc7_VOCtrain_MCGAN_300/0.mat');
    numbatch = length(fc7_features);
    m = 198*10;
    x1 = zeros(m, 4096);
    for i = 1:198
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_VOC2007_resize64/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 21
    % Test 1 - compare baseline VOCtrain 300 to gt VOC in fc7
    load('fc7_VOCtrain_baseline_300/0.mat');
    numbatch = length(fc7_features);
    m = 198*10;
    x1 = zeros(m, 4096);
    for i = 1:198
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_VOC2007_resize64/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 22
    % Test 22 - compare MCGAN NYU to gt NYU in fc7
    load('fc7_NYU_MCGAN_910/0.mat');
    numbatch = length(fc7_features);
    m = 137*10;
    x1 = zeros(m, 4096);
    for i = 1:137
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    
    n = 4600;
    x2 = zeros(n, 4096);
    for i = 0:22
       load(['fc7_NYU_test/' num2str(i)])
       inds = round(linspace(1, length(fc7_features), 200));
       for j= 1:200
          x2(j+i*200, :) = fc7_features{inds(j)}(:,5)';
       end
    end
elseif testnum == 23
    % Test 1 - compare baseline NYU to gt NYU in fc7
    load('fc7_NYU_baseline_910/0.mat');
    m = 137*10;
    x1 = zeros(m, 4096);
    for i = 1:137
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    
    n = 4600;
    x2 = zeros(n, 4096);
    for i = 0:22
       load(['fc7_NYU_test/' num2str(i)])
       inds = round(linspace(1, length(fc7_features), 200));
       for j= 1:200
          x2(j+i*200, :) = fc7_features{inds(j)}(:,5)';
       end
    end   

elseif testnum == 24 
    % Test 24 - compare MCGAN NYU to gt NYU in fc7 (epoch 1500)
    load('fc7_NYU_MCGAN_1500/0.mat');
    numbatch = length(fc7_features);
    m = 137*10;
    x1 = zeros(m, 4096);
    for i = 1:137
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    
    n = 4600;
    x2 = zeros(n, 4096);
    for i = 0:22
       load(['fc7_NYU_test/' num2str(i)])
       inds = round(linspace(1, length(fc7_features), 200));
       for j= 1:200
          x2(j+i*200, :) = fc7_features{inds(j)}(:,5)';
       end
    end
elseif testnum == 25
    % Test 25 - compare Conv MCGAN to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_Arch_Conv_30/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 26
    % Test 26 - compare Conv Leaky MCGAN to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_Arch_Leaky_30/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
elseif testnum == 27
    % Test 27 - compare Max MCGAN to gt COCO in fc7
    load('fc7_test2015/0.mat');
    numbatch = length(fc7_features);
    m = 200*10;
    x1 = zeros(m, 4096);
    for i = 1:200
        x1(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
    load('fc7_Arch_Max_30/0.mat');
    n = 200*10;
    x2 = zeros(n, 4096);
    for i = 1:200
        x2(1+(i-1)*10:i*10,:) = fc7_features{i}';
    end
end

disp('Loaded Features');
disp(size(x1));
disp(size(x2));

% Get sizes
m = size(x1, 1);
d = size(x1, 2);
n = size(x2, 1);
d_2 = size(x2, 2);
assert(d == d_2);

% Get norm matrices
% norm1 (within x1)
[idx1, idx2] = ndgrid(1:m, 1:m);
norm1 = arrayfun(@(row1, row2) norm(x1(row2,:)-x1(row1,:)), idx1, idx2);
disp('Norm1 done');

[idx1, idx2] = ndgrid(1:n, 1:n);
norm2 = arrayfun(@(row1, row2) norm(x2(row2,:)-x2(row1,:)), idx1, idx2);
disp('Norm2 done');

[idx1, idx2] = ndgrid(1:m, 1:n);
norm3 = arrayfun(@(row1, row2) norm(x2(row2,:)-x1(row1,:)), idx1, idx2);
disp('Norm3 done');
disp('Calculated norms');

% Grid search over h. Return largest MMD value
hs = [1e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 5e-1, 1, 5, 10, 50, 100, 500, 1000, 5000, 1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7, 1e8, 5e8, 1e9];
mmd2vals = zeros(size(hs));
for i = 1:length(hs)
    h = hs(i);
    sum1vals = exp(-norm1.^2/(2*h^2));
    sum1 = sum(sum(sum1vals));
    sum1 = sum1 / (m*(m-1));
    sum2vals = exp(-norm2.^2/(2*h^2));
    sum2 = sum(sum(sum2vals));
    sum2 = sum2 / (n*(n-1));
    sum3vals = exp(-norm3.^2/(2*h^2));
    sum3 = sum(sum(sum3vals));
    sum3 = (sum3*2)/(m*n);
    mmd2 = sum1 + sum2 - sum3;
    mmd2vals(i) = mmd2;
end

walltime = toc;
 
% Save to data file
save(['mmd2_test' num2str(testnum) '.mat'], 'mmd2vals');

% Display max
disp(['MMD^2 max: ' num2str(max(mmd2vals))]);

% Display how long it took
disp(['Walltime: ' num2str(walltime)]);
