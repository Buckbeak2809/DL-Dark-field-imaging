clear all;
close all;
clc
% simulate dark field image and training target with exist dataset
%% parameter setting
tic
original_pix_size = 80;
target_pix_size = 40; %nm
% pix_size_sub = 20; %nm
target_size = 256;
lambda = 532;% nm
NA_obj = 0.55;
NA_illu = 0.7;

%% Fourier filter calculation
illu_num = 30;
% illumination = zeros(target_size,target_size,illu_num);
illumination_Fourier = zeros(target_size,target_size,illu_num);
illumination_Fourier_shift = zeros(2,illu_num);
F_illu = NA_illu/lambda; % illumination F
fs = (1/target_pix_size);
f_single_pix = fs/target_size;
F_filters = zeros(target_size,target_size,illu_num);
F_cut = NA_obj/lambda;
F_target = F_cut+F_illu;

for i = 1:illu_num
    theta = 2*pi *(i/illu_num);
    x = ceil(F_illu/f_single_pix*cos(theta));
    y = ceil(F_illu/f_single_pix*sin(theta));
    illumination_Fourier_shift(:,i) = [x,y];
    illumination_Fourier(target_size/2+1+x, target_size/2+1+y, i)=1;
%     illumination(:,:,i) = ifft2(fftshift(illumination_Fourier(:,:,i)));
    F_filter_current = zeros(target_size);
    [XX,YY] = meshgrid(-target_size/2+1:target_size/2);
    rr = sqrt((XX+x).^2+(YY+y).^2);
    F_filter_current(rr<F_cut/f_single_pix)=1;
    F_filters(:,:,i) = F_filter_current;
end

F_filter_target = zeros(target_size);
F_filter_target(rr<F_target/f_single_pix)=1;

figure(1); imagesc(sum(illumination_Fourier,3)); axis image;colormap("gray")
figure(2); imagesc(sum(F_filters,3)); axis image;colormap("gray")
%% generate pairs from object ground truth
addr_read = ''; % ground truth image path 
addr_save = ''; % save path
fname = dir([addr_read '*.tif']);
img_num = length(fname);
% shuffle images
rand_ind = randperm(img_num);
i=1;
img_num_eff = img_num;
new_ind = 1;
while i<=img_num_eff 
    ind = rand_ind(i); % shuffle images
    img_gt = importdata([addr_read fname(ind).name]);
%     img_gt = fliplr(img_gt); % data augmentation
    if max(img_gt(:))<1e4
        i=i+1;
        img_num_eff=img_num_eff-1;
        continue
    end
    img_gt_FT = fftshift(fft2(img_gt));
    img_filtered = F_filters.*img_gt_FT;
    img_filtered_shiftback = img_filtered; % pre-allocation
    img_DF_sub = img_filtered_shiftback; % pre-allocation
    for j=1:illu_num
        img_filtered_shiftback(:,:,j) = imtranslate(img_filtered(:,:,j), [illumination_Fourier_shift(1,j), illumination_Fourier_shift(2,j)]);
        img_DF_sub(:,:,j) = abs(ifft2(ifftshift(img_filtered_shiftback(:,:,j)))).^2;
    end
    img_DF = sum(img_DF_sub,3);
    img_DF = mat2gray(awgn(mat2gray(img_DF),35));
    target_FT = img_gt_FT.*F_filter_target;
    img_target = mat2gray(abs(ifft2(ifftshift(target_FT))).^2);
    img_gt_out= mat2gray(img_gt);
    imwrite(img_gt_out, [addr_save 'GT\GT_' num2str(new_ind) '.tif']);
    imwrite(img_DF, [addr_save 'DF\DF_' num2str(new_ind) '.tif']);
    imwrite(img_target, [addr_save 'Target\Target_' num2str(new_ind) '.tif']);
    new_ind = new_ind+1;

    % data aug rotate 90
    img_gt = importdata([addr_read fname(ind).name]);
    img_gt = imrotate(img_gt,90); % data augmentation
    if max(img_gt(:))<1e4
        i=i+1;
        img_num_eff=img_num_eff-1;
        continue
    end
    img_gt_FT = fftshift(fft2(img_gt));
    img_filtered = F_filters.*img_gt_FT;
    img_filtered_shiftback = img_filtered; % pre-allocation
    img_DF_sub = img_filtered_shiftback; % pre-allocation
    for j=1:illu_num
        img_filtered_shiftback(:,:,j) = imtranslate(img_filtered(:,:,j), [illumination_Fourier_shift(1,j), illumination_Fourier_shift(2,j)]);
        img_DF_sub(:,:,j) = abs(ifft2(ifftshift(img_filtered_shiftback(:,:,j)))).^2;
    end
    img_DF = sum(img_DF_sub,3);
    img_DF = mat2gray(awgn(mat2gray(img_DF),35));
    target_FT = img_gt_FT.*F_filter_target;
    img_target = mat2gray(abs(ifft2(ifftshift(target_FT))).^2);
    img_gt_out = mat2gray(img_gt);
    imwrite(img_gt_out, [addr_save 'GT\GT_' num2str(new_ind) '.tif']);
    imwrite(img_DF, [addr_save 'DF\DF_' num2str(new_ind) '.tif']);
    imwrite(img_target, [addr_save 'Target\Target_' num2str(new_ind) '.tif']);
    new_ind = new_ind+1;

    % data aug rotate 180
    img_gt = importdata([addr_read fname(ind).name]);
    img_gt = imrotate(img_gt,180); % data augmentation
    if max(img_gt(:))<1e4
        i=i+1;
        img_num_eff=img_num_eff-1;
        continue
    end
    img_gt_FT = fftshift(fft2(img_gt));
    img_filtered = F_filters.*img_gt_FT;
    img_filtered_shiftback = img_filtered; % pre-allocation
    img_DF_sub = img_filtered_shiftback; % pre-allocation
    for j=1:illu_num
        img_filtered_shiftback(:,:,j) = imtranslate(img_filtered(:,:,j), [illumination_Fourier_shift(1,j), illumination_Fourier_shift(2,j)]);
        img_DF_sub(:,:,j) = abs(ifft2(ifftshift(img_filtered_shiftback(:,:,j)))).^2;
    end
    img_DF = sum(img_DF_sub,3);
    img_DF = mat2gray(awgn(mat2gray(img_DF),35));
    target_FT = img_gt_FT.*F_filter_target;
    img_target = mat2gray(abs(ifft2(ifftshift(target_FT))).^2);
    img_gt_out = mat2gray(img_gt);
    imwrite(img_gt_out, [addr_save 'GT\GT_' num2str(new_ind) '.tif']);
    imwrite(img_DF, [addr_save 'DF\DF_' num2str(new_ind) '.tif']);
    imwrite(img_target, [addr_save 'Target\Target_' num2str(new_ind) '.tif']);
    new_ind = new_ind+1;

    % data aug flil left right
    img_gt = importdata([addr_read fname(ind).name]);
    img_gt = fliplr(img_gt); % data augmentation
    if max(img_gt(:))<1e4
        i=i+1;
        img_num_eff=img_num_eff-1;
        continue
    end
    img_gt_FT = fftshift(fft2(img_gt));
    img_filtered = F_filters.*img_gt_FT;
    img_filtered_shiftback = img_filtered; % pre-allocation
    img_DF_sub = img_filtered_shiftback; % pre-allocation
    for j=1:illu_num
        img_filtered_shiftback(:,:,j) = imtranslate(img_filtered(:,:,j), [illumination_Fourier_shift(1,j), illumination_Fourier_shift(2,j)]);
        img_DF_sub(:,:,j) = abs(ifft2(ifftshift(img_filtered_shiftback(:,:,j)))).^2;
    end
    img_DF = sum(img_DF_sub,3);
    img_DF = mat2gray(awgn(mat2gray(img_DF),35));
    target_FT = img_gt_FT.*F_filter_target;
    img_target = mat2gray(abs(ifft2(ifftshift(target_FT))).^2);
    img_gt_out= mat2gray(img_gt);
    imwrite(img_gt_out, [addr_save 'GT\GT_' num2str(new_ind) '.tif']);
    imwrite(img_DF, [addr_save 'DF\DF_' num2str(new_ind) '.tif']);
    imwrite(img_target, [addr_save 'Target\Target_' num2str(new_ind) '.tif']);
    new_ind = new_ind+1;
    
    i=i+1;
%     figure(3); imagesc(abs(img_gt)); axis image; colormap("gray")
%     figure(4); imagesc(abs(img_DF_sub(:,:,1))); axis image; colormap("gray")
%     figure(5); imagesc(abs(img_DF)); axis image; colormap("gray")
%     figure(6); imagesc(log(abs(target_FT))); axis image; colormap("gray")
%     figure(7); imagesc(abs(img_target)); axis image; colormap("gray")
end
toc