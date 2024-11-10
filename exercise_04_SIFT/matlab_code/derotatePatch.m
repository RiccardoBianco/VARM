function derotated_patch = derotatePatch(img, loc, patch_size, ori)

% The patch can overlap at most sqrt(2)/2 * patch_size over the image edge.
% To prevent this, pad the image with zeros
% TODO: Your code here

% compute derotated patch  
for px=1:patch_size
    for py=1:patch_size
% TODO: Your code here

      % rotate patch by angle ori
% TODO: Your code here

      % move coordinates to patch
% TODO: Your code here

      % sample image (using nearest neighbor sampling as opposed to more
      % accuracte bilinear sampling)
% TODO: Your code here
    end
end
      
end

