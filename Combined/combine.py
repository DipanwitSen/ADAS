import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import ultralytics as YOLO

# Global variables
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

# Dictionary mapping class IDs to class names
classes = {1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle'}

def measure_curvature_real(left_fit, right_fit, ploty):
    y_eval = np.max(ploty)
    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    return left_curverad, right_curverad

def calculate_time_to_collision(object_distance, relative_velocity):
    if relative_velocity <= 0:
        return float('inf')  # No collision if objects are not moving towards each other
    return object_distance / relative_velocity

def calculate_steering_angle(left_fit, right_fit, ploty):
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
    steering_angle = np.arctan((left_curverad + right_curverad) / 2)
    return steering_angle

def detect_obstacles(binary_warped):
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1000
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary_warped)
    return keypoints

def draw_lane_lines(image, binary_warped, left_fit, right_fit, Minv, curvature, offset, steering_angle, obstacles, velocity, acceleration, brake, drive_mode, lane_keeping, detected_objects, kalman_predictions, collision_warnings):
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    left_curvature, right_curvature = curvature
    curvature_text = "Curvature: {:.2f} m^-1, {:.2f} m^-1".format(left_curvature, right_curvature)
    cv2.putText(result, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    steering_angle_text = "Steering Angle: {:.2f} rad".format(steering_angle)
    cv2.putText(result, steering_angle_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    if len(obstacles) > 0:
        obstacles_text = "Obstacles Detected: {}".format(len(obstacles))
        cv2.putText(result, obstacles_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    offset_text = "Vehicle Offset: {:.2f} m".format(offset)
    cv2.putText(result, offset_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    velocity_text = "Velocity: {:.2f} km/h".format(velocity)
    cv2.putText(result, velocity_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    acceleration_text = "Acceleration: {:.2f} m/s^2".format(acceleration)
    cv2.putText(result, acceleration_text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    brake_text = "Brake: {}".format("ON" if brake else "OFF")
    cv2.putText(result, brake_text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    drive_mode_text = "Drive Mode: {}".format(drive_mode)
    cv2.putText(result, drive_mode_text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    lane_keeping_text = "Lane Keeping: {}".format(lane_keeping)
    cv2.putText(result, lane_keeping_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    for (bbox, label), pred in zip(detected_objects, kalman_predictions):
        x, y, w, h = bbox
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        pred_x, pred_y = int(pred[0][0]), int(pred[1][0])
        cv2.circle(result, (pred_x, pred_y), 5, (0, 255, 255), -1)
        cv2.putText(result, "Prediction", (pred_x, pred_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        warning_y = 500
        for label, ttc in collision_warnings:
            warning_text = f"WARNING: {label} - Time to collision: {ttc:.2f}s"
            cv2.putText(result, warning_text, (50, warning_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            warning_y += 50

    return result

def calculate_vehicle_offset(image, left_fit, right_fit):
    height = image.shape[0]
    lane_center = (left_fit[0] * height ** 2 + left_fit[1] * height + left_fit[2] +
                   right_fit[0] * height ** 2 + right_fit[1] * height + right_fit[2]) / 2
    car_center = image.shape[1] / 2
    offset = (car_center - lane_center) * xm_per_pix  # xm_per_pix should be defined globally
    return offset

def get_perspective_transform(src_points, dst_points):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    return M, Minv

def calculate_velocity(left_fit, right_fit, ploty):
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    velocity = 30 * np.cos(steering_angle)  # assume a constant velocity of 30 km/h
    return velocity

def calculate_acceleration(left_fit, right_fit, ploty):
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    acceleration = 2 * np.sin(steering_angle)  # assume a constant acceleration of 2 m/s^2
    return acceleration

def calculate_brake(left_fit, right_fit, ploty):
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    brake = np.abs(steering_angle) > 0.5  # assume braking if steering angle is greater than 0.5 radians
    return brake

def warp_perspective(image, M, img_size):
    return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

def calculate_drive_mode(left_fit, right_fit, ploty):
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    drive_mode = "AUTO" if np.abs(steering_angle) < 0.5 else "MANUAL"
    return drive_mode

def calculate_lane_keeping(left_fit, right_fit, ploty):
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
    lane_keeping = "GOOD" if np.abs(left_curverad - right_curverad) < 100 else "BAD"
    return lane_keeping

def canny(image):
    return cv2.Canny(image, 50, 150)

def sliding_window_polyfit(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base= np.argmax(histogram[midpoint:]) + midpoint
    nwindows = 9
    margin = 100
    minpix = 50
    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                           (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit, out_img

def extract_detected_objects(yolo, image):
    objects = []
    results = yolo(image)

    for result in results:
        boxes = result.boxes.cpu().numpy()
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = box.conf[0]
            cls = int(box.cls[0])

            if conf > 0.5 and cls in [2, 3, 5, 7]:  # car, motorbike, bus, truck
                label = result.names[cls]
                objects.append(((int(x1), int(y1), int(x2 - x1), int(y2 - y1)), label))

    return objects

def run_kalman_filter(detected_objects, prev_kalman_predictions, prev_kalman_covariances, transition_matrix, observation_matrix, process_noise_cov, observation_noise_cov):
    kalman_predictions = []
    kalman_covariances = []

    for (bbox, label), prev_pred, prev_cov in zip(detected_objects, prev_kalman_predictions, prev_kalman_covariances):
        measurement = np.array([[bbox[0]], [bbox[1]]])
        predicted_state = np.dot(transition_matrix, prev_pred)
        predicted_covariance = np.dot(np.dot(transition_matrix, prev_cov), transition_matrix.T) + process_noise_cov

        innovation = measurement - np.dot(observation_matrix, predicted_state)
        innovation_covariance = np.dot(np.dot(observation_matrix, predicted_covariance), observation_matrix.T) + observation_noise_cov
        kalman_gain = np.dot(np.dot(predicted_covariance, observation_matrix.T), np.linalg.inv(innovation_covariance))

        updated_state = predicted_state + np.dot(kalman_gain, innovation)
        updated_covariance = predicted_covariance - np.dot(np.dot(kalman_gain, observation_matrix), predicted_covariance)

        kalman_predictions.append(updated_state)
        kalman_covariances.append(updated_covariance)

    return kalman_predictions, kalman_covariances

# Constants for Kalman filter
transition_matrix = np.array([[1, 0], [0, 1]])
observation_matrix = np.array([[1, 0], [0, 1]])
process_noise_cov = np.array([[1, 0], [0, 1]])
observation_noise_cov = np.array([[1, 0], [0, 1]])

# Initialize previous Kalman filter states
prev_kalman_predictions = [np.array([[0], [0]])] * 10
prev_kalman_covariances = [np.eye(2)] * 10

def process_image(image, yolo, src_points, dst_points):
    binary_warped = canny(image)
    left_fit, right_fit, out_img = sliding_window_polyfit(binary_warped)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])
    curvature = measure_curvature_real(left_fit, right_fit, ploty)
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    offset = calculate_vehicle_offset(image, left_fit, right_fit)
    obstacles = detect_obstacles(binary_warped)
    detected_objects = extract_detected_objects(yolo, image)
    velocity = calculate_velocity(left_fit, right_fit, ploty)
    acceleration = calculate_acceleration(left_fit, right_fit, ploty)
    brake = calculate_brake(left_fit, right_fit, ploty)
    drive_mode = calculate_drive_mode(left_fit, right_fit, ploty)
    lane_keeping = calculate_lane_keeping(left_fit, right_fit, ploty)
    M, Minv = get_perspective_transform(src_points, dst_points)
    binary_warped = warp_perspective(binary_warped, M, (image.shape[1], image.shape[0]))
    kalman_predictions, kalman_covariances = run_kalman_filter(detected_objects, prev_kalman_predictions,
                                                               prev_kalman_covariances, transition_matrix,
                                                               observation_matrix, process_noise_cov,
                                                               observation_noise_cov)

    # Calculate time to collision for each detected object
    collision_warnings = []
    for (bbox, label), pred in zip(detected_objects, kalman_predictions):
        object_distance = bbox[1]  # Assuming y-coordinate represents distance
        relative_velocity = velocity - pred[1][0]  # Assuming pred[1][0] is object's velocity
        time_to_collision = calculate_time_to_collision(object_distance, relative_velocity)

        if time_to_collision <= 5:  # 5 seconds threshold
            collision_warnings.append((label, time_to_collision))

    result = draw_lane_lines(image, binary_warped, left_fit, right_fit, Minv, curvature, offset, steering_angle,
                             obstacles, velocity, acceleration, brake, drive_mode, lane_keeping, detected_objects,
                             kalman_predictions, collision_warnings)
    return result

def process_video(video_path, yolo, src_points, dst_points):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        result = process_image(frame, yolo, src_points, dst_points)
        out.write(result)

        cv2.imshow('Output', result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    choice = input("Choose input type (image, video, webcam): ")

    # Load YOLO model
    yolo = YOLO.YOLO('yolov8n.pt')

    if choice == 'image':
        image_path = input("Enter the path to the image: ")
        if not os.path.exists(image_path):
            print("Image file not found.")
            return
        image = cv2.imread(image_path)
        src_points = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
        dst_points = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
        result = process_image(image, yolo, src_points, dst_points)
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == 'video':
        video_path = input("Enter the path to the video: ")
        if not os.path.exists(video_path):
            print("Video file not found.")
            return
        src_points = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
        dst_points = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
        process_video(video_path, yolo, src_points, dst_points)
    elif choice == 'webcam':
        src_points = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
        dst_points = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
        process_webcam(yolo, src_points, dst_points)
    else:
        print("Invalid choice. Please choose 'image', 'video', or 'webcam'.")

if __name__ == "__main__":
    main()
