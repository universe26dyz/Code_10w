function imshow3(img, range, layout)
    if nargin < 2
        range = [];
    end
    if nargin < 3
        layout = [];
    end
    
    [nx, ny, nz] = size(img);
    
    % 如果提供了layout参数，调整显示布局
    if ~isempty(layout) && length(layout) == 2
        rows = layout(1);
        cols = layout(2);
        % 确保rows*cols >= nz
        if rows*cols < nz
            error('Layout size is too small for the number of images');
        end
    else
        % 默认布局
        cols = nz;
        rows = 1;
    end
    
    % 创建一个拼接图像
    montage_img = zeros(nx*rows, ny*cols);
    for i = 1:nz
        row_idx = floor((i-1)/cols) + 1;
        col_idx = mod(i-1, cols) + 1;
        row_range = (row_idx-1)*nx + 1 : row_idx*nx;
        col_range = (col_idx-1)*ny + 1 : col_idx*ny;
        montage_img(row_range, col_range) = img(:,:,i);
    end
    
    imshow(montage_img, range);
end