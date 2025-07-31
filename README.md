# Video Classifier
A simple tool that detects whether a video contains more than one person.
Built using YOLOv8n and the OpenCV Python library.

# Required packages
All required packages are listed in `requirements.txt`.

# Setup Instructions

`python3 -m venv video_classifier_env`
`source video_classifier_env/bin/activate`

`python3 -m pip install --upgrade pip`
`python3 -m pip install -r requirements.txt`

# Running the Classifier

Open a Python shell:
`python3`

Then run the following commands:
`from video_classifier import Video_Classifier`

`video_class = Video_Classifier(output_frames_fl=True, path_to_output_folder='frames_output_data')`

`# Optional: run validation using labeled data`
`video_class.run_validation(path_to_labels='input_data/labels.txt', path_to_input_folder='input_data/videos')`

`# Run the classifier`
`video_class.run(path_to_input_folder='input_data/videos')`

To exit:
`quit()`
`deactivate`

# Notes

- All paths should be provided without a trailing '/'.
- The labels.txt file (used for validation) must contain two columns:
  - video: Name of the video file without the .mp4 extension
  - label: 0 for one or fewer people in the video, 1 for more than one person
