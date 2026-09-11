%% Copyright reserved by Zhenfeng Lyu, modified by Honzhang Huang
close all; clc; clear
addpath(genpath('C:\Users\user\Desktop\MultiMapCode')); % Please choose your own address

fpath='DYL_20260521_212145\MM_sax_801\'; % Please choose your own file

%% 1. load dicom images and info
all_dcm=dir([fpath,'*.dcm']);
for kdcm=1:size(all_dcm)
    Mag(:,:,kdcm) = double(dicomread([fpath,all_dcm(kdcm).name]));
    Info = dicominfo([fpath,all_dcm(kdcm).name]);
    Acq_time(kdcm) = HHMMSS2Sec(Info.AcquisitionTime)*1e3;
end
Info=dicominfo([fpath,all_dcm(1).name]);

% sort 
[Xsorted,I] = sort(Acq_time,'ascend');
Ysorted = zeros(size(Mag));
Acq_time_sorted = zeros(size(Acq_time));
for kmag=1:size(I,2)
    Ysorted(:,:,kmag) = Mag(:,:,I(kmag));
    Acq_time_sorted(kmag) = Acq_time(I(kmag));
end
Mag = Ysorted;
Acq_time = Acq_time_sorted;

%% 2. Crop image to reduce mapping consumption
[Nx,Ny,Nz] = size(Mag);
NumImg = 10; % number of images per slice
NumSlice = Nz/NumImg;
Mag = reshape(Mag, [Nx,Ny,NumImg,NumSlice]);
Acq_time = reshape(Acq_time, [NumImg,NumSlice]);
Acq_time_check = round(Acq_time_sorted(2:end)-Acq_time_sorted(1:end-1))';

figure;
imshow(Mag(end/4:end/4*3,end/4:end/4*3,1,1),[])
Mag_crop = Mag(end/4:end/4*3,end/4:end/4*3,:,:);

figure;
for kmag=1:size(Mag_crop,4)
    nexttile,
    imshow3(Mag_crop(:,:,:,kmag),[])
end

%% 3. Registration
MIND_mag_reg = zeros(size(Mag_crop));
MIND_mag_reg(:,:,1,:) = Mag_crop(:,:,1,:);
disp('Begin registration');
for kmag=1:size(Mag_crop,4)
    disp(['Slice',num2str(kmag),' +++++++++++++++++']);
    tic
    img = squeeze(Mag_crop(:,:,:,kmag));
    fixedImage = single(squeeze(img(:,:,1)));   % 目标图像
    parfor k=2:size(img,3)
        inputImage = single(squeeze(img(:,:,k)));   % 发生变形的图像
        [u1,v1,movingImage] = deformableReg2Dmind_asym_nodisplay(fixedImage,inp utImage,.3);
        MIND_mag_reg(:,:,k,kmag) = movingImage;
    end
    toc
end

%% save
TR = Info.RepetitionTime;
VPS = Info.EchoTrainLength;
FA = [45, 45, 45];
TI = [50, 150];
T2_prep=[35, 45, 55];

save('data.mat','Mag','Mag_crop','Info','Acq_time','TR','VPS','T2_prep','FA','TI')
