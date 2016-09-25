import sys, re, imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def readDetections(filename):
    
    detRE = re.compile('([0-9.]+) - ([0-9.]+) s, ([0-9]+)x([0-9]+) - ([0-9]+)x([0-9]+)')
    detections = []
    with open(filename) as f:
        for line in f:
            match = detRE.match(line)
            if match:
                detections.append(((float(match.group(1)), int(match.group(3)), int(match.group(4))), (float(match.group(2)), int(match.group(5)), int(match.group(6)))))
    return detections


def drawDetections(videoIn, videoOut, detections, color = (255, 0, 0), cellSize = 16):
    
    font = ImageFont.truetype('arial.ttf', 28)
    
    reader = imageio.get_reader(videoIn, 'ffmpeg')
    
    fps = reader.get_meta_data()['fps']
    detections = [((int(a[0] * fps), a[1] * cellSize, a[2] * cellSize), (int(b[0] * fps), (b[1] + 1) * cellSize, (b[2] + 1) * cellSize)) for a, b in detections]
    
    writer = imageio.get_writer(videoOut, 'ffmpeg', fps = fps, macro_block_size = 8)
    
    try:
        for num, frame in enumerate(reader):
            try:
                for numDet, (a, b) in enumerate(detections):
                    if (num >= a[0]) and (num <= b[0]):
                        img = Image.fromarray(frame)
                        draw = ImageDraw.Draw(img)
                        draw.rectangle([a[1], a[2], b[1], b[2]], outline = color )
                        draw.text([a[1] + 10, a[2] + 10], str(numDet + 1), fill = color, font = font)
                        frame = np.array(img)
                        del draw, img
                writer.append_data(frame)
            except Exception as e:
                print(str(e))
    except:
        pass
    
    reader.close()
    writer.close()


if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print('Usage: {} <detections-file> <video-in> <video-out> [<num-detections> [<cell-size = 16>]]'.format(sys.argv[0]))
        exit()
    
    det = readDetections(sys.argv[1])
    if len(sys.argv) > 4:
        det = det[:int(sys.argv[4])]
    cs = int(sys.argv[5]) if len(sys.argv) > 5 else 16
    
    drawDetections(sys.argv[2], sys.argv[3], det, cellSize = cs)