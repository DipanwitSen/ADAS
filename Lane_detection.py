import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

def get_perspective_transform(src_points, dst_points):
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    Minv = cv2.getPerspectiveTransform(dst_points, src_points)
    return M, Minv

def warp_perspective(image, M, img_size):
    return cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)

def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([
        [(200, height), (1100, height), (550, 250)]
    ])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, (0, 255, 0))  # Green color mask
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def sliding_window_polyfit(binary_warped):
    histogram = np.sum(binary_warped[binary_warped.shape[0]//2:,:], axis=0)
    out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255

    midpoint = int(histogram.shape[0]//2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    nwindows = 9
    margin = 100
    minpix = 50

    window_height = int(binary_warped.shape[0]//nwindows)
    nonzero = binary_warped.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    leftx_current = leftx_base
    rightx_current = rightx_base

    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        win_y_low = binary_warped.shape[0] - (window+1)*window_height
        win_y_high = binary_warped.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0,255,0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0,255,0), 2)

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

def measure_curvature_real(left_fit_cr, right_fit_cr, ploty):
    ym_per_pix = 30/720 # meters per pixel in y dimension
    xm_per_pix = 3.7/700 # meters per pixel in x dimension

    y_eval = np.max(ploty)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    return left_curverad, right_curverad

def calculate_vehicle_offset(image, left_fit, right_fit):
    xm_per_pix = 3.7 / 700  # meters per pixel in x dimension
    car_position = image.shape[1] / 2
    height = image.shape[0]
    bottom_left = left_fit[0] * height**2 + left_fit[1] * height + left_fit[2]
    bottom_right = right_fit[0] * height**2 + right_fit[1] * height + right_fit[2]
    lane_center_position = (bottom_left + bottom_right) / 2
    center_offset_meters = (car_position - lane_center_position) * xm_per_pix
    return center_offset_meters

def draw_lane_lines(original_img, binary_img, left_fit, right_fit, Minv):
    ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
    left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    newwarp = cv2.warpPerspective(color_warp, Minv, (original_img.shape[1], original_img.shape[0]))
    result = cv2.addWeighted(original_img, 1, newwarp, 0.3, 0)
    return result

def process_image(image_path, src_points, dst_points, output_dir):
    image = cv2.imread(image_path)
    img_size = (image.shape[1], image.shape[0])

    M, Minv = get_perspective_transform(src_points, dst_points)
    warped = warp_perspective(image, M, img_size)
    
    canny_image = canny(warped)
    left_fit, right_fit, out_img = sliding_window_polyfit(canny_image)
    
    result = draw_lane_lines(image, canny_image, left_fit, right_fit, Minv)
    offset = calculate_vehicle_offset(image, left_fit, right_fit)
    
    ploty = np.linspace(0, canny_image.shape[0]-1, canny_image.shape[0])
    left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
    
    output_image_path = os.path.join(output_dir, "processed_image.jpg")
    cv2.imwrite(output_image_path, result)

    plt.imshow(result)
    plt.title(f'Offset: {offset:.2f} meters, Curvature: Left = {left_curverad:.2f} m, Right = {right_curverad:.2f} m')
    plt.show()

def process_video(video_source, src_points, dst_points, output_dir):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {video_source}")
        return

    img_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    M, Minv = get_perspective_transform(src_points, dst_points)
    output_video_path = os.path.join(output_dir, "processed_video.avi")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 20.0, img_size)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        warped = warp_perspective(frame, M, img_size)
        canny_image = canny(warped)
        left_fit, right_fit, out_img = sliding_window_polyfit(canny_image)
        
        result = draw_lane_lines(frame, canny_image, left_fit, right_fit, Minv)
        offset = calculate_vehicle_offset(frame, left_fit, right_fit)
        
        ploty = np.linspace(0, canny_image.shape[0]-1, canny_image.shape[0])
        left_curverad, right_curverad = measure_curvature_real(left_fit, right_fit, ploty)
        
        cv2.putText(result, f'Offset: {offset:.2f} meters, Curvature: Left = {left_curverad:.2f} m, Right = {right_curverad:.2f} m', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        out.write(result)
        cv2.imshow("result", result)
        
        if cv2.waitKey(1) == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

def main():
    src_points = np.float32([
        [585, 460],
        [203, 720],
        [1127, 720],
        [695, 460]
    ])
    dst_points = np.float32([
        [320, 0],
        [320, 720],
        [960, 720],
        [960, 0]
    ])
    
    output_dir = "C:\\Users\\KIIT\\Desktop\\intel\\lane_detection\\store"
    os.makedirs(output_dir, exist_ok=True)

    user_choice = input("Enter 'image' to process an image or 'video' to process a video: ").strip().lower()
    if user_choice == 'image':
        image_path = input("Enter the path to the image: ").strip()
        process_image(image_path, src_points, dst_points, output_dir)
    elif user_choice == 'video':
        video_choice = input("Enter 'webcam' to use webcam or 'file' to provide a video file path: ").strip().lower()
        if video_choice == 'webcam':
            process_video(0, src_points, dst_points, output_dir)
        elif video_choice == 'file':
            video_path = input("Enter the path to the video file: ").strip()
            if os.path.exists(video_path):
                process_video(video_path, src_points, dst_points, output_dir)
            else:
                print(f"Error: The file {video_path} does not exist.")
        else:
            print("Invalid choice. Please enter 'webcam' or 'file'.")
    else:
        print("Invalid choice. Please enter 'image' or 'video'.")

if __name__ == "__main__":
    main()

