
# lane-pothole-detection

## Class Variables

- `MASK_THRESHOLD` (float): The threshold for masking the top portion of the image, i.e. masking the sky out. (default is 4.5/8).
- `LEN_THRESHOLD` (int): The minimum contour length to be considered for lane detection (default is 100).

## Member Functions

### `preprocess_image(self, img: np.ndarray) -> np.ndarray`
Basic thresholding and preprocessing of the image.

- **Parameters**:
  - `img` (numpy.ndarray): Image received from the camera.

- **Returns**:
  - `numpy.ndarray`: Preprocessed image (after grayscaling, binary thresholding, blurring).
  
---

### `get_contours(self, binary_img: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]`
Identifies the contours of the potholes and lanes from the binary image.

- **Parameters**:
  - `binary_img` (numpy.ndarray): Image received after preprocessing.

- **Returns**:
  - `lane_contours` (List[numpy.ndarray]): Contours detected for lanes.
  - `pothole_contours` (List[numpy.ndarray]): Contours detected for potholes.

---

### `draw_contours(self, shape: Tuple[int, int, int], contours: Union[np.ndarray, List[np.ndarray]]) -> np.ndarray`
Draws the specified contours on a blank image.

- **Parameters**:
  - `shape` (Tuple[int, int, int]): Shape of the output image (height, width, channels).
  - `contours` (Union[np.ndarray, List[np.ndarray]]): Contours to be drawn on the image.

- **Returns**:
  - `img` (numpy.ndarray): Image with drawn contours.

---

### `get_lane_pothole_images(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]`
Runs contour detection on the camera image and returns the contours on a black background.

- **Parameters**:
  - `img` (numpy.ndarray): Image received from the camera.

- **Returns**:
  - `lane_image` (numpy.ndarray): Image with lane contours drawn.
  - `pothole_image` (numpy.ndarray): Image with pothole contours drawn.

---

### `camera_callback(self, data: Image) -> None`
Processes the camera input and publishes lane and pothole images to the respective ROS topics.

- **Parameters**:
  - `data` (Image): ROS message containing the camera image.

- **Returns**:
  - `None`
  
---

## Usage Notes
- Modify the class variables as needed
- Change the ROS topics of the publishers/subscribers as needed
- Pothole detection condition inside get_contours may need tuning
