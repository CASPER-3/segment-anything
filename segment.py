from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import numpy as np

# import torch
import matplotlib.pyplot as plt
import cv2

sam_checkpoint = "checkpoint/sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

img_pth = "assets/"


def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)
    


def show_max_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    ann = anns[0]
    m = ann["segmentation"]
    print(m.shape)
    color_mask = np.concatenate([np.random.random(3), [0.35]])
    img[m] = color_mask
    ax.imshow(img)


def show_max_n_anns(anns, n):
    if len(anns) == 0:
        return
    if len(anns) < n:
        n = len(anns)

    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for i in range(n):
        ann = anns[i]
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)


def show_max_area_anns(anns, area,image):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        m_area = ann["area"]
        if m_area < area:
            break
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
        print("area is ",m_area)
        calc_boundary(m,image)
    ax.imshow(img)

def calc_boundary(segmentation,image):
    # mask_area = np.asarray(segmentation == True).nonzero()
    # print(mask_area)
    col,row = np.where(segmentation)
    # print(col)
    # print(row)
    # print(col.shape)
    # print(row.shape)
    area_size = col.shape[0]
    points = []
    for i in range(area_size):
        points.append([row[i],col[i]])
    points_arr = np.array(points)
    # print(points_arr)
    find_boundary(points_arr,image)
    image_save = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite("result/boundary.png",image_save)
    
def find_boundary(points,image):
    # 找到x坐标最小和最大的点
    min_x_point = points[np.argmin(points[:, 0])]
    max_x_point = points[np.argmax(points[:, 0])]
    # 在这两个点中找到y坐标最大和最小的点
    top_left = min_x_point if min_x_point[1] > max_x_point[1] else max_x_point
    bottom_right = max_x_point if top_left is min_x_point else min_x_point
    # 在剩下的点中找到y坐标最大和最小的点
    remaining_points = points[(points != top_left).any(axis=1) & (points != bottom_right).any(axis=1)]
    top_right = remaining_points[np.argmax(remaining_points[:, 1])]
    bottom_left = remaining_points[np.argmin(remaining_points[:, 1])]
    print('Top left:', top_left)
    print('Top right:', top_right)
    print('Bottom left:', bottom_left)
    print('Bottom right:', bottom_right)
    cv2.circle(image, (top_left[0], top_left[1]), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(image, (top_right[0], top_right[1]), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(image, (bottom_left[0], bottom_left[1]), radius=5, color=(255, 0, 0), thickness=-1)
    cv2.circle(image, (bottom_right[0], bottom_right[1]), radius=5, color=(255, 0, 0), thickness=-1)
    

    


def segment_scene(img_name):
    image = cv2.imread(img_pth + img_name)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(image)
    print(len(masks))
    sorted_anns = sorted(masks, key=(lambda x: x["area"]), reverse=True)
    with open("result/" + img_name + "_masks.json", "w") as f:
        for item in sorted_anns:
            f.write(str(item) + "\n")
        f.close()

    plt.figure(figsize=(20, 20))
    plt.imshow(image)
    show_max_area_anns(masks, 30000,image)
    plt.axis("off")
    plt.savefig("result/" + img_name + "_segment.png")


segment_scene("0.jpg")
