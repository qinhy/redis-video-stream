class PatchableImage:
    def __init__(self,img_file,patch_size=(512,512)):
        self.patch_size = patch_size
        self.img_file = img_file
        self.img_raw = Image.open(img_file)
        self.img_raw_shape = self.img_raw.size[::-1]
        assert len(self.img_raw_shape) == 3 or len(self.img_raw_shape) == 2
        self.img = self._add_margin(self.img_raw, 0, patch_size[1] - self.img_raw_shape[1]%patch_size[1], patch_size[0] - self.img_raw_shape[0]%patch_size[0],0)
        self.ndimg = np.asarray(self.img)[:,:,:3]
        self.patches, self.patches_merge_func = self.split_image_into_patches(self.ndimg, self.ndimg.shape[0]//patch_size[0], self.ndimg.shape[1]//patch_size[1])
    
    def get_merge_imgs(self, imgs=None):
        if imgs is None:
            imgs = self.patches
        res = self.patches_merge_func(imgs)
        res = Image.fromarray(res[:self.img_raw_shape[0],:self.img_raw_shape[1]])
        assert res.size == self.img_raw.size
        return res

    def _add_margin(self, pil_img, top, right, bottom, left, color=(0,0,0)):
        width, height = pil_img.size
        new_width = width + right + left
        new_height = height + top + bottom
        result = Image.new(pil_img.mode, (new_width, new_height), color)
        result.paste(pil_img, (left, top))
        return result
    
    def merge_patches(self, patches, rows, cols):
        # Reshape the flat list of patches into a list of rows
        patches_rows = [patches[i*cols:(i+1)*cols] for i in range(rows)]
        # For each row, concatenate the patches horizontally
        rows_images = [np.hstack(row_patches) for row_patches in patches_rows]
        # Concatenate the rows vertically and return the result
        return np.vstack(rows_images)

    def split_image_into_patches(self, image, n, m):
        assert image.shape[0]%n==0 and image.shape[1]%m==0,f'{image.shape}%({n,m})!=0'
        patches = []
        # Split the image into n-sized chunks along the rows
        for sub_image in np.array_split(image, n):
            # Split each chunk into m-sized chunks along the columns
            for patch in np.array_split(sub_image, m, axis=1):
                patches.append(patch)
        return patches,lambda x:self.merge_patches(x, rows=n, cols=m)



    # def apply_label_split(self,idx,labels):
    #     self.pcds[idx].set_labels(np.expand_dims(labels,0).T)
    #     ul = np.unique(labels)
    #     for j in ul:
    #         self.label_splits.setdefault(j, [])
    #         self.label_splits[j].append(self.pcds[idx].select_by_bool(labels==j))
    #     return self.label_splits

    # def __len__(self):
    #     return sum(list(map(lambda x:x.size(),self.pcds)))
        



