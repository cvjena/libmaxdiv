Video Anomaly Detection Experiment
==================================

This directory contains code used to produce the results reported in section 4.4 of the paper.


Preparing the Data
------------------

The video used for this experiment can be downloaded from the ViSOR repository here: [http://www.openvisor.org/video_details.asp?idvideo=339][1]

It consists of 1495 frames which we have extracted as individual JPEG files using [ffmpeg][2].

Image features have been extracted for each frame from the `conv5` layer of [CaffeNet][3] and reduced to 16 dimensions using PCA. The features must be stored in a numpy file as a 5-dimensional array, whose first dimensions is time, the second and third dimensions are x and y coordinates, respectively, the fourth dimension is a dummy dimension of size 1, and the last dimensions refers to feature channels.

The resulting numpy dump of the video features can be downloaded [here][4] (68 MB).


Detecting Anomalies
-------------------

The python script `visor_detect.py` can be used to detect anomalies in the video based on the features extracted before.

The argument `--divergence` can be used to switch between unbiased KL divergence and cross entropy.

Processing the entire video using a full scan would take a few hours. This can be reduced to a couple of minutes using interval proposals by passing the flag `--proposals`.

For example, to detect anomalies using cross entropy as divergence measure, one could use this:

    python visor_detect.py visor_cnn_16.npy --divergence CROSS_ENTROPY

Expected result:

    33.8 - 36.8 s, 7x0 - 16x4 (Score: 13.51977371799444)
    32.1 - 35.6 s, 12x7 - 21x11 (Score: 11.027415414724373)
    2.7 - 6.6 s, 0x6 - 10x10 (Score: 9.372703441161079)
    3.4 - 6.4 s, 12x7 - 21x11 (Score: 4.24910010887028)
    32.5 - 36.1 s, 0x6 - 9x10 (Score: -4.508607601136234)

With the `--proposals` flag, the results would look like this after just 5 minutes of computation on 8 virtual cores:

    34.0 - 38.8 s, 12x9 - 21x13 (Score: 7.463871368981771)
    2.8 - 6.5 s, 0x6 - 9x10 (Score: 6.2026769016791405)
    2.4 - 6.0 s, 11x7 - 20x11 (Score: -2.1263733154340656)
    32.6 - 35.9 s, 0x6 - 9x10 (Score: -10.571929907412255)
    20.2 - 23.8 s, 8x6 - 17x10 (Score: -27.53710660633071)

Unfortunately, the original top-1 detection (the cyclist on the sidewalk) is missing, but the remaining detections are very similar.
It gets detected at position 9, though, which is not included in the list shown here.


Visualizing the Results
-----------------------

The provided scripts `draw_video_detections.py` and `draw_video_frames.py` can be used to generate a video or individual frames onto which the detections are drawn as bounding boxes.

The detections returned by `visor_detect.py` need to be stored in a file that is passed as argument to these scripts. The path to the original video needs to be specified as well.

For example, to generate a video with detections, one would use:

    python draw_video_detections.py detections.txt visor_1261565143617_cesta1.wmv detections.avi

Sampling individual frames with the top 5 detections every 4 seconds could be achieved using the following:

    python draw_video_frames.py detection_frames/ 5 4 visor_1261565143617_cesta1.wmv detections.txt

| Unbiased KL Divergence | Cross Entropy |
|:----------------------:|:-------------:|
| ![Unbiased KL Divergence][5] | ![Cross Entropy][6] |



[1]: http://www.openvisor.org/video_details.asp?idvideo=339
[2]: http://ffmpeg.org/
[3]: https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet
[4]: https://drive.google.com/open?id=1QSF0rC6zFt4fyGvpbD18Bh2JTtmOtEmB
[5]: https://user-images.githubusercontent.com/7915048/42508044-79f6567e-8447-11e8-9267-4ef861e8a948.gif
[6]: https://user-images.githubusercontent.com/7915048/42508043-79dcb32c-8447-11e8-90da-ff524182aed9.gif