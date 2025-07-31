import os
import numpy as np
import pandas as pd

import cv2
from ultralytics import YOLO
from imutils.object_detection import non_max_suppression
from skimage.feature import local_binary_pattern

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score

class Video_Classifier:


  def __init__(self, output_frames_fl=False, path_to_output_folder='outputs'):

      self.output_frames_fl = output_frames_fl
      self.path_to_output_folder = path_to_output_folder
      os.makedirs(self.path_to_output_folder, exist_ok=True)

      ### Initiating models:
      self.person_classifier = YOLO('yolov8n.pt') # Using a pre-trained YOLO (nano) for people detection.
      self.face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

      ### Results storage:
      self.validation_total_results = []
      self.df_validation_total_results = pd.DataFrame()


  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


  def run(self, path_to_input_folder='input_data/videos'):

    videofiles_list = [f for f in os.listdir(path_to_input_folder) if f.lower().endswith(('.mp4', '.mov', '.avi', '.mkv', '.flv'))]
    
    if len(videofiles_list) > 0:

      ### Iterating over all videos in the provided input path:
      for video in np.sort(videofiles_list):

        self.video_file = video.split('.')[0]

        print(f'Processing video {self.video_file}...')

        self.video_filename = f'{video}'
        self.path_to_video = path_to_input_folder+f'/{self.video_filename}'

        try:
          pred_label, spoofed_flag, count_detected_people, manual_validation_flag = self.analyse_video()

          if pred_label == 0:
            print('The video does not have more than one person in it.')
          elif pred_label == 1:
            print('The video has more than one person in it.')
          else:
            print('Could not determine result.')

          if manual_validation_flag:
            print('The video requires manual validation due to low certainty of people detections.')

          if spoofed_flag == 1 and count_detected_people == 0:
            print('The video has only a spoof attempt (document/picture presented) but no person was detected.')

        except:
          print('Failed to process video.')

        print('\n')

    else:

      print('No videos located at the provided path.')


  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


  '''
  Model validation.
  '''

  def run_validation(self, path_to_labels='input_data/labels.txt', path_to_input_folder='input_data/videos'):

    ### Reading the labels. Structure: video, label.
    df_labels = pd.read_csv(path_to_labels, sep=None, engine='python')

    ### Iterating over all videos in the provided labels dataset:
    for index, row in df_labels.iterrows():

      self.video_file = row['video']
      self.true_label = row['label']

      print(f'Processing video {self.video_file}...')

      self.video_filename = f'{self.video_file}.mp4'
      self.path_to_video = path_to_input_folder+f'/{self.video_filename}'

      try:
        pred_label, spoofed_flag, count_detected_people, manual_validation_flag = self.analyse_video()

        ### Summary of the video:
        self.validation_total_results.append((self.video_file, self.true_label, pred_label, spoofed_flag, count_detected_people, manual_validation_flag))
        # print(self.video_file, self.true_label, pred_label, spoofed_flag, count_detected_people, manual_validation_flag)
      
      except:
        print('Failed to process video.')

      print('\n')

    cv2.destroyAllWindows()
    self.df_validation_total_results = pd.DataFrame(self.validation_total_results, columns=['video', 'true_label', 'pred_label', 'spoofed_flag', 'count_detected_people', 'manual_validation_flag'])


    print('Confusion Matrix:')
    print(confusion_matrix(self.df_validation_total_results['true_label'], self.df_validation_total_results['pred_label'], labels=[0,1]))

    accuracy = accuracy_score(self.df_validation_total_results['true_label'], self.df_validation_total_results['pred_label'])
    print(f"Accuracy: {accuracy:.4f}")

    precision = precision_score(self.df_validation_total_results['true_label'], self.df_validation_total_results['pred_label'], pos_label=1)
    print(f"Precision (suspicious - 2+ people in the video): {precision:.4f}")

    recall = recall_score(self.df_validation_total_results['true_label'], self.df_validation_total_results['pred_label'], pos_label=1)
    print(f"Recall (suspicious - 2+ people in the video): {recall:.4f}")


    self.video_file = None; self.true_label = None; self.path_to_video = None



  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


  def analyse_video(self):

    pred_label = 0 ### Are there >1 people in the video?
    spoofed_flag = np.nan ### Flagging whether there was a spoofing (presentation) attack attempt.
    manual_validation_flag = np.nan ### Flagging whether any manual validation is necessary in case of low certainty of detections.

    cap = cv2.VideoCapture(self.path_to_video)

    ### Saving potential spoofing attempts:
    spoof_flags_list = []
    spoof_scores_list = []

    ### Lists for tracking detections per frame:
    people_count_list = []
    detection_scores_list = []

    ### Iterating over the frames of the video:
    frame_count = 0
    while cap.isOpened():

      ret_frame, frame = cap.read()
      if not ret_frame:
          break


      '''
      Detecting people in the frame via YOLO.
      '''

      ### Resizing the image with preserving the original aspect ratio:
      frame = self.resize_with_aspect_ratio(frame, width=640)

      ### Ensuring a 3-channel image for YOLO:
      if len(frame.shape) == 2 or frame.shape[2] == 1:
          frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

      ### Applying YOLO detection to detect people (and other classes):
      frame_model = frame.copy()
      results_i = self.person_classifier(frame_model, verbose=False)[0]

      boxes = results_i.boxes.xyxy.cpu().numpy()
      scores = results_i.boxes.conf.cpu().numpy()
      classes = results_i.boxes.cls.cpu().numpy()

      ### Filtering for person boxes only (class 0 in YOLO, with probability of >=50%):
      person_boxes = []
      person_probs = []
      for i in range(len(boxes)):
          if int(classes[i]) == 0 and scores[i] >= 0.50:
              person_boxes.append(boxes[i])
              person_probs.append(scores[i])


      '''
      Detecting faces for each detected person and applying NMS to reduce the number of faces to the unique numbers.
      '''

      face_crops_global = []
      face_boxes_global = []

      for j, (x1, y1, x2, y2) in enumerate(person_boxes):

          person_crop = frame[int(y1):int(y2), int(x1):int(x2)] ### A bounding box of the detected person.


          ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###

          ### Detecting face of the person:
          gray_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
          faces = self.face_detector.detectMultiScale(gray_crop, scaleFactor=1.1, minNeighbors=7)

          ### If no face found, then use the full person box - to still detect as much as possible:
          if len(faces) == 0:

            face_crops_global.append(person_crop)
            face_boxes_global.append([int(x1), int(y1), int(x2), int(y2)])

          else:

            face_boxes = []
            for (xf, yf, wf, hf) in faces:
                face_boxes.append([xf, yf, xf + wf, yf + hf])

            # Run NMS on face detections to remove multiple overlaps for the same person if any - a lower overlap to make sure that the face is detected fully within the person box.
            # Although it is possible that YOLO captures >1 person in the same box.
            face_boxes_nms = non_max_suppression(np.array(face_boxes), overlapThresh=0.40)

            for (xf1, yf1, xf2, yf2) in face_boxes:

                face_crop = person_crop[int(yf1):int(yf2), int(xf1):int(xf2)]
                face_crops_global.append(face_crop)

                face_boxes_global.append([int(xf1 + x1), int(yf1 + y1), int(xf2 + x1), int(yf2 + y1)]) ### Global frame coordinates.

          ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


      face_boxes_global = np.array(face_boxes_global)

      ### Applying NMS with a high threshold on faces to remove multiple overlapping detections of the same face:
      face_boxes_nms = non_max_suppression(face_boxes_global, overlapThresh=0.90)

      ### Saving indices of the face detections kept:
      face_indices_retained = []
      for nbox in face_boxes_nms:
          for idx, obox in enumerate(face_boxes_global):
              if np.allclose(nbox, obox):
                  face_indices_retained.append(idx)
                  break


      '''
      Adding detections to the resized frame (after identifying is any are spoofed).
      Saving detections and probabilties to a list - to later estimate the number of people over a set of frames.
      '''

      frame_count_people = 0
      frame_detection_score = 0
      frame_count_spoofs = 0
      frame_spoof_score = 0
      for i in face_indices_retained:
          face_crop = face_crops_global[i]
          face_box = face_boxes_global[i]

          ### Spoof detection via static features:
          is_real, spoof_score = self.detect_spoof(face_crop)
          if not is_real:
            frame_count_spoofs += 1
            frame_spoof_score += spoof_score

          ### Adding label to the resized frame:
          face_x1, face_y1, face_x2, face_y2 = face_box

          color = (0, 255, 0) if is_real else (0, 0, 255)
          spoof_label = f'REAL ({person_probs[j]*100:.1f}%)' if is_real else f'SPOOF ({spoof_score*100:.1f}%)'
          cv2.rectangle(frame, (face_x1, face_y1), (face_x2, face_y2), color, 2)
          cv2.putText(frame, spoof_label, (face_x1, face_y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

          ### Adding to counts if not a spoof:
          if is_real:
            frame_count_people += 1
            frame_detection_score += person_probs[j]

      people_count_list.append(frame_count_people)
      if frame_count_people > 0:
        detection_scores_list.append(1.0 * frame_detection_score / frame_count_people) ### Appending average score over all non-spoof people detected.
      else:
        pass

      if frame_count_spoofs > 0:
        spoof_flags_list.append(False)
        spoof_scores_list.append(1.0 * frame_spoof_score / frame_count_spoofs) ### Appending average score over all spoof attempts on the same frame.


      '''
      Estimating number of people in the video.
      Assuming 30 FPS - detecting average no. of people over 30 frames ~ each consequtive 1 second of the video.
      It is strict but need to ensure that a person is not forced to pass identification (by capturing another person invideo even for 1 second).
      '''

      if (frame_count > 0) and (frame_count % 30 == 0):

        ### If observed >1.5 person on average over 1 second, then there were more than 1 people in the video during 1 second:
        if len(people_count_list) > 0:
          avg_detected_people = np.mean(people_count_list[-30:])

          if avg_detected_people > 1.5:
            pred_label = 1
        else:
          pass

        ### Validating if there was an attempt of spoofing:
        if len(spoof_scores_list) > 0:
          avg_spoof_score = np.mean(spoof_scores_list[-30:])

          if spoofed_flag == 1:
            pass
          else:
            spoofed_flag = int((avg_spoof_score >= 0.5) and (len(spoof_scores_list[-30:]) / 30 >= 0.5)) if spoof_flags_list else -1 ### if spoof is detected in at least 50% of frames with a high score
        else:
          spoofed_flag = 0

        ### Validating if manual validation is necessary due to insufficient scores of person detection:
        if len(detection_scores_list) > 0:
          avg_detection_score = np.mean(detection_scores_list[-30:])

          if manual_validation_flag == 1:
            pass
          else:
            manual_validation_flag = int(avg_detection_score < 0.85) if detection_scores_list else -1 ### Business rule: <85% probaivlity of detection requires manual verification.
        else:
          manual_validation_flag = -1


      '''
      Saving the frame.
      '''

      if self.output_frames_fl:

        video_output_data_path = self.path_to_output_folder+'/'+self.video_filename.split('.')[0]
        os.makedirs(video_output_data_path, exist_ok=True)

        frame_filename = os.path.join(video_output_data_path, f'frame_{frame_count:03d}.jpg')
        cv2.imwrite(frame_filename, frame)


      frame_count += 1


    cap.release()


    '''
    In case the video is shorter than 30 frames:
    '''

    if frame_count <= 30:

        ### If observed >1.5 person on average over ~1 second, then there were more than 1 people in the video during 1 second:
        if len(people_count_list) > 0:
          avg_detected_people = np.mean(people_count_list[-frame_count-1:])

          if avg_detected_people > 1.5:
            pred_label = 1
        else:
          pass

        ### Validating if there was an attempt of spoofing:
        if len(spoof_scores_list) > 0:
          avg_spoof_score = np.mean(spoof_scores_list[-frame_count-1:])

          if spoofed_flag == 1:
            pass
          else:
            spoofed_flag = int((avg_spoof_score >= 0.5) and (len(spoof_scores_list[-frame_count-1:]) / (frame_count-1) >= 0.5)) if spoof_flags_list else -1 ### if spoof is detected in at least 50% of frames with a high score
        else:
          spoofed_flag = -1

        ### Validating if manual validation is necessary due to insufficient scores of person detection:
        if len(detection_scores_list) > 0:
          avg_detection_score = np.mean(detection_scores_list[-frame_count-1:])

          if manual_validation_flag == 1:
            pass
          else:
            manual_validation_flag = int(avg_detection_score < 0.85) if detection_scores_list else -1 ### Business rule: <85% probaivlity of detection requires manual verification.
        else:
          manual_validation_flag = -1


    '''
    Summarising video information.
    '''

    ### If the video was spoofed (i.e., a document was presented as per requirement), was the person present on the video as well?
    if (spoofed_flag == 1) and (pred_label == 0) and (len(people_count_list) > 0):
      count_detected_people = np.round(np.mean(people_count_list))
    else:
      count_detected_people = -1


    '''
    Returning the result.
    '''

    return pred_label, spoofed_flag, count_detected_people, manual_validation_flag


  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


  '''
  Resizing the frame with keeping the aspect ratio.
  '''

  def resize_with_aspect_ratio(self, image, width=None, height=None, inter=cv2.INTER_AREA):

      (h, w) = image.shape[:2]
      if width is None and height is None:
          return image

      if width is not None:
          r = width / float(w)
          dim = (width, int(h * r))
      else:
          r = height / float(h)
          dim = (int(w * r), height)

      return cv2.resize(image, dim, interpolation=inter)


  ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


  '''
  Detecting image spoofing using a simpler methods (instead of DL models).
  '''

  def detect_spoof(self, face_crop, lbp_top_k_bins_thresh=3.0, lbp_peak_ratio_thresh=0.25, color_entropy_thresh=5.5, edge_density_thresh=0.05, lap_thresh=60):

      results = {}

      face_crop_gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
      h, w = face_crop_gray.shape
      face_crop_gray_center = face_crop_gray[int(h*0.10):int(h*0.90), int(w*0.10):int(w*0.90)]  ### Taking a cenral part of the face crop (to remove extra noise).


      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
      ### Texture check - LBP (Local Binary Patterns) histogram:
      radius = 2; n_points = 8 * radius
      lbp = local_binary_pattern(face_crop_gray_center, n_points, radius, method='uniform')

      # Normalised histogram:
      hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points+3), range=(0, n_points+2))
      hist = hist.astype('float'); hist /= (hist.sum() + 1e-6)

      # Count of bins of  high (>10%) frequency - dominating patterns:
      lbp_top_k_bins = np.sum(hist > 0.10)

      results['lbp_real_1'] = lbp_top_k_bins >= lbp_top_k_bins_thresh ### higher for real faces since real faces have more diverse histograms

      # Peak ratio - the most dominating pattern:
      lbp_peak_ratio = np.max(hist)
      results['lbp_real_2'] = lbp_peak_ratio < lbp_peak_ratio_thresh ### checking if there is no one dominating pattern - not expected for real faces
      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
      ### Color histogram flateness check - real faces have more varied colors:
      hist = cv2.calcHist([face_crop], [0], None, [32], [0, 256])
      hist = cv2.normalize(hist, hist).flatten()
      color_entropy = -np.sum(hist * np.log2(hist + 1e-8))
      results['color_real'] = color_entropy > color_entropy_thresh
      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
      ### Checking for presence of edges - real faces have more detail (e.g., wrinkles):
      edges = cv2.Canny(face_crop_gray, 100, 200)
      edge_density = np.mean(edges > 0)
      results['edge_real'] = edge_density > edge_density_thresh
      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
      ### Texture check - Laplacian variance:
      lap_var =  cv2.Laplacian(face_crop_gray_center, cv2.CV_64F).var()
      results['lap_real'] = lap_var >= lap_thresh ### NB! lower sharpness is expected if the face is moving (motion blur due to movement ==> Laplacian variance is lower)
      ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###


      ### Final decision:
      real__votes = int(results['lbp_real_1'] or results['lbp_real_2']) + int(results['color_real']) + int(results['edge_real']) + int(results['lap_real'])
      is_real = real__votes >= 3

      ### In case there's only 1 dominating pattern, then this is most definitely a spoof:
      if lbp_top_k_bins == 1:
        real__votes = 0
        is_real = False

      spoof_score = 1 - real__votes / 4.0


      return is_real, spoof_score
