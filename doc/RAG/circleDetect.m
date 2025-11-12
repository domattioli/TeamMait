I = imread('L06 sunflower.png');
I = imresize(I,0.4);
I_gray = im2gray(I);
I_canny = edge(I_gray,'Canny');
[h,w] = size(I_canny);
radii = 20:2:80;
acc = zeros(h,w,length(radii));
[y_idx,x_idx] = find(I_canny);
for k = 1:length(radii)
    r = radii(k);
    for i = 1:length(x_idx)
        x = x_idx(i);
        y = y_idx(i);
        for theta = 0:359
            a = round(x - r*cosd(theta));
            b = round(y - r*sind(theta));
            if a>0 && a<=w && b>0 && b<=h
                acc(b,a,k) = acc(b,a,k) + 1;
            end
        end
    end
end
[m,ind] = max(acc(:));
[b,a,k] = ind2sub(size(acc),ind);
imshow(I);
hold on;
viscircles([a b],radii(k),'Color','r');
hold off;