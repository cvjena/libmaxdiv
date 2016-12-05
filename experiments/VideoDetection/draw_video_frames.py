""" Draws detection bounding boxes onto a video and exports frames as still images. """

import sys, os, re, imageio
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


def drawDetections(videoIn, outDir, detections, intvl = 4, colors = [(255, 0, 0), (0, 0, 255), (0, 255, 0)], cellSize = 16):
    
    font = ImageFont.truetype('arial.ttf', 36)
    
    reader = imageio.get_reader(videoIn, 'ffmpeg')
    
    fps = reader.get_meta_data()['fps']
    intvl = int(intvl * fps)
    detections = [
            [((int(a[0] * fps), a[1] * cellSize, a[2] * cellSize), (int(b[0] * fps), (b[1] + 1) * cellSize, (b[2] + 1) * cellSize)) for a, b in det]
        for det in detections]
    
    try:
        for num, frame in enumerate(reader):
            if num % intvl == 0:
                try:
                    img = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img)
                    for det, col in zip(detections, colors):
                        for numDet, (a, b) in enumerate(det):
                            if (num >= a[0]) and (num <= b[0]):
                                for offs in range(-4, 5, 1):
                                    draw.rectangle(
                                        [max(0, a[1] + offs), max(0, a[2] + offs), min(img.size[0], b[1] - offs), min(img.size[1], b[2] - offs)],
                                        outline = (255, 255, 255) if abs(offs) > 2 else col
                                    )
                                for xoffs in range(-1, 2, 1):
                                    for yoffs in range(-1, 2, 1):
                                        draw.text(
                                            [a[1] + 10 + xoffs, a[2] + 10 + yoffs],
                                            str(numDet + 1), fill = (255, 255, 255), font = font
                                        )
                                draw.text([a[1] + 10, a[2] + 10], str(numDet + 1), fill = col, font = font)
                    img.save(os.path.join(outDir, 'frame{:04d}.jpg'.format(num)), quality = 100)
                    del draw, img
                except Exception as e:
                    print(str(e))
    except:
        pass
    
    reader.close()


if __name__ == '__main__':
    
    if len(sys.argv) < 5:
        print('Usage: {} <out-dir> <num-detections> <intvl> <video-in> <detections-file>+'.format(sys.argv[0]))
        exit()
    
    outDir = sys.argv[1]
    numDet = int(sys.argv[2])
    intvl = float(sys.argv[3])
    videoIn = sys.argv[4]
    det = [readDetections(sys.argv[i])[:numDet] for i in range(5, len(sys.argv))]
    
    drawDetections(videoIn, outDir, det, intvl)