import timm
from timm.models.vision_transformer import VisionTransformerPEG, VisionTransformer
import torch
import matplotlib.pyplot as plt


kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)
peg_idxs = [-1, 0, 1]
# where to insert position encoding generator,
# if it is -1, it is inserted before the first block
# multiple position encoding generator can be inserted simultaneously
peg_model = timm.create_model("deit3_base_patch16_224", pretrained=True, img_size=384)
peg_model.eval()
img = torch.zeros(4, 3, 384, 384)
out = peg_model(img)
print(f"input size: {img.shape}")
print(f"output size: {out.shape}")
print(peg_model.pos_embed.squeeze(0).shape)
corr = torch.corrcoef(peg_model.pos_embed.squeeze(0))
#plt.imshow(corr_1.detach().numpy())
#plt.savefig('figures/corr.png')

n_patches=24
f, axarr = plt.subplots(n_patches,n_patches, figsize=(20,20))
# for i in range(1,197):
#     axarr[int((i-1)/14),(i-1)%14].imshow(corr[i][1:].reshape(14,14).detach().numpy())
#     axarr[int((i-1)/14),(i-1)%14].axis('off')

for i in range(0,n_patches**2):
    axarr[int((i)/n_patches),(i)%n_patches].imshow(corr[i].reshape(n_patches,n_patches).detach().numpy())
    axarr[int((i)/n_patches),(i)%n_patches].axis('off')

f.suptitle('Positional Encoddings Correlation', fontsize=20)
f.supylabel('Input patch row', fontsize=20)
f.supxlabel('Input patch column', fontsize=20)
plt.savefig('figures/pos_embedding_viz_deit_highres.png')

#print(timm.list_models('*deit*'))
