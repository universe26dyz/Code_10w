%%% This file is to merge image frames on one image
function Merge_Imgs = ImgMerge(Imgs, Msize)
ImgSiz = size(Imgs);

if prod(ImgSiz(3:end))~=prod(Msize)
    error('ImgMerge: Merge size is wrong.')
end

Merge_Imgs = zeros(ImgSiz(1)*Msize(1),ImgSiz(2)*Msize(2));
for ky=1:Msize(1)
    for kx=1:Msize(2)
        Merge_Imgs(1+(ky-1)*ImgSiz(1):ky*ImgSiz(1),1+(kx-1)*ImgSiz(2):kx*ImgSiz(2)) = Imgs(:,:,kx+(ky-1)*Msize(2));
    end
end

end
