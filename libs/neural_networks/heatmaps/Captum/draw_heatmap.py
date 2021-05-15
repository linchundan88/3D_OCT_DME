
'''


for i in range(pred_class_num):
    # predict_max_class = attributions[1][0][i]
    attribution1 = shap_values_results[0][i]

    # attributions.shape: (1, 299, 299, 3)
    data = attribution1[0]
    data = np.mean(data, -1)

    abs_max = np.percentile(np.abs(data), 100)
    abs_min = abs_max

    # dx, dy = 0.05, 0.05
    # xx = np.arange(0.0, data1.shape[1], dx)
    # yy = np.arange(0.0, data1.shape[0], dy)
    # xmin, xmax, ymin, ymax = np.amin(xx), np.amax(xx), np.amin(yy), np.amax(yy)
    # extent = xmin, xmax, ymin, ymax

    # cmap = 'RdBu_r'
    # cmap = 'gray'
    cmap = 'seismic'
    plt.axis('off')
    # plt.imshow(data1, extent=extent, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
    # plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)

    # fig = plt.gcf()
    # fig.set_size_inches(2.99 / 3, 2.99 / 3)  # dpi = 300, output = 700*700 pixels
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)

    if blend_original_image:
        plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        save_filename1 = list_images[i]
        plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
        plt.close()

        img_heatmap = cv2.imread(list_images[i])
        (tmp_height, tmp_width) = img_original.shape[:-1]
        img_heatmap = cv2.resize(img_heatmap, (tmp_width, tmp_height))
        img_heatmap_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.jpg'.format(i))
        cv2.imwrite(img_heatmap_file, img_heatmap)

        dst = cv2.addWeighted(img_original, 0.65, img_heatmap, 0.35, 0)
        img_blend_file = os.path.join(os.path.dirname(list_images[i]), 'deepshap_blend_{0}.jpg'.format(i))
        cv2.imwrite(img_blend_file, dst)

        # region create gif
        import imageio

        mg_paths = [img_original_file, img_heatmap_file, img_blend_file]
        gif_images = []
        for path in mg_paths:
            gif_images.append(imageio.imread(path))
        img_file_gif = os.path.join(os.path.dirname(list_images[i]), 'deepshap_{0}.gif'.format(i))
        imageio.mimsave(img_file_gif, gif_images, fps=gif_fps)
        list_images[i] = img_file_gif
        # endregion
    else:
        plt.imshow(data, interpolation='none', cmap=cmap, vmin=-abs_min, vmax=abs_max)
        save_filename1 = list_images[i]
        plt.savefig(save_filename1, bbox_inches='tight', pad_inches=0)
        plt.close()



'''


