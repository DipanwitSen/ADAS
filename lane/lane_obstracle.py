import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Global variables
ym_per_pix = 30 / 720  # meters per pixel in y dimension
xm_per_pix = 3.7 / 700  # meters per pixel in x dimension

def measure_curvature_real(left_fit, right_fit, ploty):
    y_eval = np.max(ploty)

    left_curverad = ((1 + (2 * left_fit[0] * y_eval * ym_per_pix + left_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * left_fit[0])
    right_curverad = ((1 + (2 * right_fit[0] * y_eval * ym_per_pix + right_fit[1]) ** 2) ** 1.5) / np.absolute(
        2 * right_fit[0])

    return left_curverad, right_curverad


def calculate_steering_angle(left_fit, right_fit, ploty):
    # Calculate the steering angle based on the lane curvature
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
    steering_angle = np.arctan((left_curverad + right_curverad) / 2)
    return steering_angle


def detect_obstacles(binary_warped):
    # Use a blob detector to find obstacles in the binary warped image
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10
    params.maxThreshold = 200
    params.filterByArea = True
    params.minArea = 100
    params.maxArea = 1000
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(binary_warped)
    return keypoints


def draw_lane_lines(image, binary_warped, left_fit, right_fit, Minv, curvature, offset, steering_angle, obstacles, velocity, acceleration, brake, drive_mode, lane_keeping):
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

    # Display curvature
    left_curvature, right_curvature = curvature
    curvature_text = "Curvature: {:.2f} m^-1, {:.2f} m^-1".format(left_curvature, right_curvature)
    cv2.putText(result, curvature_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display steering angle
    steering_angle_text = "Steering Angle: {:.2f} rad".format(steering_angle)
    cv2.putText(result, steering_angle_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display detected obstacles
    if len(obstacles) > 0:
        obstacles_text = "Obstacles Detected: {}".format(len(obstacles))
        cv2.putText(result, obstacles_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display vehicle offset
    offset_text = "Vehicle Offset: {:.2f} m, {:.2f} m".format(offset[0], offset[1])
    cv2.putText(result, offset_text, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display velocity
    velocity_text = "Velocity: {:.2f} km/h".format(velocity)
    cv2.putText(result, velocity_text, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display acceleration
    acceleration_text = "Acceleration: {:.2f} m/s^2".format(acceleration)
    cv2.putText(result, acceleration_text, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display brake status
    brake_text = "Brake: {}".format("ON" if brake else "OFF")
    cv2.putText(result, brake_text, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display drive mode
    drive_mode_text = "Drive Mode: {}".format(drive_mode)
    cv2.putText(result, drive_mode_text, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display lane keeping status
    lane_keeping_text = "Lane Keeping: {}".format(lane_keeping)
    cv2.putText(result, lane_keeping_text, (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

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
    # Calculate the velocity based on the lane curvature and steering angle
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    velocity = 30 * np.cos(steering_angle)  # assume a constant velocity of 30 km/h
    return velocity


def calculate_acceleration(left_fit, right_fit, ploty):
    # Calculate the acceleration based on the lane curvature and steering angle
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    acceleration = 2 * np.sin(steering_angle)  # assume a constant acceleration of 2 m/s^2
    return acceleration


def calculate_brake(left_fit, right_fit, ploty):
    # Calculate the brake status based on the lane curvature and steering angle
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    brake = np.abs(steering_angle) > 0.5  # assume braking if steering angle is greater than 0.5 radians
    return brake


def warp_perspective(image, M, img_size):
    return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)


def calculate_drive_mode(left_fit, right_fit, ploty):
    # Calculate the drive mode based on the lane curvature and steering angle
    steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
    drive_mode = "auto" if np.abs(steering_angle) < 0.5 else "manual"
    return drive_mode


def calculate_lane_keeping(left_fit, right_fit, ploty):
    # Calculate the lane keeping status based on the lane curvature and steering angle
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
    lane_keeping = "good" if np.abs(left_curverad - right_curverad) < 100 else "bad"
    return lane_keeping


def canny(image):
    return cv2.Canny(image, 50, 150)


def sliding_window_polyfit(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0] // 2:, :], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255

    midpoint = int(histogram.shape[0] // 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    window_height = int(binary_warped.shape[0] // nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    leftx_current = leftx_base
    rightx_current = rightx_base

    margin = 100
    minpix = 50

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window + 1) * window_height
        win_y_high = binary_warped.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low),
                      (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low),
                      (win_xright_high, win_y_high), (0, 255, 0), 2)

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


def process_image(image_path, src_points, dst_points, output_dir):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Cannot open or find the image {image_path}. Check the file path.")

    M, Minv = get_perspective_transform(src_points, dst_points)
    binary_warped = canny(image)
    binary_warped = warp_perspective(binary_warped, M, (image.shape[1], image.shape[0]))

    left_fit, right_fit, out_img =sliding_window_polyfit(binary_warped)
    ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

    curvature = measure_curvature_real(left_fit, right_fit, ploty)
    offset = calculate_vehicle_offset(image, left_fit, right_fit)
    result = draw_lane_lines(image, binary_warped, left_fit, right_fit, Minv)

    # Save the processed image
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    filename = os.path.basename(image_path)
    output_path = os.path.join(output_dir, filename)
    cv2.imwrite(output_path, result)

    return curvature, offset, result


def process_video(video_path, src_points, dst_points, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open or find the video {video_path}. Check the file path.")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, 'output.mp4'), fourcc, 20.0, (1280, 720))

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            M, Minv = get_perspective_transform(src_points, dst_points)
            binary_warped = canny(frame)
            binary_warped = warp_perspective(binary_warped, M, (frame.shape[1], frame.shape[0]))

            left_fit, right_fit, out_img = sliding_window_polyfit(binary_warped)
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

            curvature = measure_curvature_real(left_fit, right_fit, ploty)
            offset = calculate_vehicle_offset(frame, left_fit, right_fit)
            steering_angle = calculate_steering_angle(left_fit, right_fit, ploty)
            obstacles = detect_obstacles(binary_warped)
            velocity = calculate_velocity(left_fit, right_fit, ploty)
            acceleration = calculate_acceleration(left_fit, right_fit, ploty)
            brake = calculate_brake(left_fit, right_fit, ploty)
            drive_mode = calculate_drive_mode(left_fit, right_fit, ploty)
            lane_keeping = calculate_lane_keeping(left_fit, right_fit, ploty)

            result = draw_lane_lines(frame, binary_warped, left_fit, right_fit, Minv, curvature, offset, steering_angle, obstacles, velocity, acceleration, brake, drive_mode, lane_keeping)
            out.write(result)
            cv2.imshow('Processed Video', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def process_webcam(src_points, dst_points, output_dir):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open webcam")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(os.path.join(output_dir, 'webcam_output.mp4'), fourcc, 20.0, (640, 480))

    while True:
        ret, frame = cap.read()
        if ret:
            M, Minv = get_perspective_transform(src_points, dst_points)
            binary_warped = canny(frame)
            binary_warped = warp_perspective(binary_warped, M, (frame.shape[1], frame.shape[0]))

            left_fit, right_fit, out_img = sliding_window_polyfit(binary_warped)
            ploty = np.linspace(0, binary_warped.shape[0] - 1, binary_warped.shape[0])

            result = draw_lane_lines(frame, binary_warped, left_fit, right_fit, Minv)
            out.write(result)
            cv2.imshow('Webcam Lane Detection', result)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    src_points = np.float32([[200, 720], [1100, 720], [595, 450], [685, 450]])
    dst_points = np.float32([[300, 720], [980, 720], [300, 0], [980, 0]])
    output_dir = 'output'

    print("Select input type:")
    print("1. Image")
    print("2. Video")
    print("3. Webcam")
    choice = int(input("Enter choice (1/2/3): "))

    if choice == 1:
        image_path = input("Enter the image path: ")
        curvature, offset, processed_image = process_image(image_path, src_points, dst_points, output_dir)
        print(f'Curvature: {curvature}')
        print(f'Offset: {offset}')
        plt.imshow(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        plt.show()
    elif choice == 2:
        if choice == 2:
            video_path = input("Enter the video path: ")
            process_video(video_path, src_points, dst_points, output_dir)
    elif choice == 3:
        process_webcam(src_points, dst_points, output_dir)
    else:
        print("Invalid choice. Please restart and select a valid option.")
