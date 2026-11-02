%% Copyright reserved by Zhenfeng Lyu, modified by Honzhang Huang
close all; clc; clear
addpath(genpath('C:\Users\user\Desktop\MultiMapCode')); % Please choose your own address

load("ColormapT1MRF.mat")
load("ColormapT2MRF.mat")

%% load data
load('data.mat')
imgs_all = Mag_crop;

%% build Tlist
T1_list = [20:20:500 505:5:1500 1520:20:2500];
T2_list = [5:5:100 110:10:200];
B1_list = 0.1:0.05:1.2;
% B1_list = 1;

T1_l = length(T1_list);
T2_l = length(T2_list);
B1_l = length(B1_list);

Tlist = zeros(T1_l*T2_l*B1_l, 5);
idx = 1;

tic
for ind = 1:T1_l
    for jnd = 1:T2_l
        for bnd = 1:B1_l
            if T1_list(ind) > T2_list(jnd)
                Tlist(idx,1) = T1_list(ind);
                Tlist(idx,2) = T2_list(jnd);
                Tlist(idx,3) = B1_list(bnd);
                idx = idx + 1;
            end
        end
    end
end
toc

Tlist(all(Tlist == 0,2),:) = [];
Tlist = unique(Tlist,'rows');

%% matching
tt = tic;
for kslice=1:size(imgs_all,4)
    [T1mapTmp, T2mapTmp, B1mapTmp] = function_T1T2_10HB_bssfp(imgs_all(:,:,:,kslice),Info,Acq_time(:,kslice),FA,Tlist,T2_prep,TI);
    T1map(:,:,:,kslice) = T1mapTmp;
    T2map(:,:,:,kslice) = T2mapTmp;
    B1map(:,:,:,kslice) = B1mapTmp;
end
fprintf("Matching consumption is %0.2f seconds.", toc(tt))
map = cat(3, T1map, T2map, B1map);

save('map.mat','map')

%% show maps
load("ColormapT1MRF.mat")
load("ColormapT2MRF.mat")
load("map.mat")

T1_range = [500,2500];
T2_range = [0,120];
B1_range = [0.4,1.2];

ImgAlign = [2,7];

f3 = figure('Name','T1 T2 B1 Maps');
subplot(3,1,1)
imshow(ImgMerge(squeeze(map(:,:,1,:)),ImgAlign), T1_range, 'Colormap', ColormapT1)
colorbar
title('T1 Map')

subplot(3,1,2)
imshow(ImgMerge(squeeze(map(:,:,2,:)),ImgAlign), T2_range, 'Colormap', ColormapT2)
colorbar
title('T2 Map')

subplot(3,1,3)
imshow(ImgMerge(squeeze(map(:,:,3,:)),ImgAlign), B1_range, 'Colormap', jet(128))
colorbar
title('B1 Map')
