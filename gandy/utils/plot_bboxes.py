import matplotlib.pyplot as plt
import matplotlib.patches as patches

def plot_bboxes(image, boxes):
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    ax.imshow(image)

    # Each box is in the form (top_left_x, top_left_y, top_right_x, bottom_left_x)
    for box in boxes:
        x = box[0]
        y = box[1]
        width = box[2] - box[0]
        height = box[3] - box[1]

        rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    
    plt.show()
