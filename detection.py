#MTCNN
import mtcnn
from matplotlib import pyplot
from matplotlib.patches import Rectangle
from mtcnn.mtcnn import MTCNN

# pyplot.interactive(False)

def draw_boxes(filename, result_list):
    data = pyplot.imread(filename)

    pyplot.imshow(data)

    ax = pyplot.gca()

    print(ax)

    for result in result_list:
        x, y, width, height = result['box']
        rect = Rectangle((x, y), width, height, fill=False, color='blue')
        ax.add_patch(rect)
    pyplot.show()


filename = "/Users/SPA/PycharmProjects/Young/archive/train/ben_afflek/httpcsvkmeuaeccjpg.jpg"

pixels = pyplot.imread(filename)

detector = MTCNN()
faces = detector.detect_faces(pixels)

draw_boxes(filename, faces)


