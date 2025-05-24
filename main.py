import cv2
import time
from ultralytics import solutions


    
def main():
    camera = cv2.VideoCapture(0)
    assert camera.isOpened(), 'Error open camera'
    time_duration = 5
    last_action_time = time.time()
    time_since_last_action = 0
    # Video writer
    w, h, fps = (int(camera.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))


    gym = solutions.AIGym(
        show=True,
        model='yolo11n-pose.pt',
        kpts=[10,8,6],
        down_angle=125.0 # For push up
    )

    while camera.isOpened():
        success, push_up = camera.read()

        if not success:
            print('Video frame is empty or processing is complete')

        results = gym(push_up)
        print(results)
        workout_stage = results.workout_stage
        totals_push_up = results.workout_count[0]
        """
            If stage when push up is up -> then start calculate 
            If time >= 2 then break and end the session
        """
        if all (stage == 'up' for stage in workout_stage):
            current_time = time.time()
            time_since_last_action = current_time - last_action_time
            if time_since_last_action > time_duration:
                print(f'Timeout, stop count {totals_push_up}')
                break
        else :
            last_action_time = time.time()
            time_since_last_action = 0 
        #cv2.imshow('Push up detection',push_up)

    camera.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    main()
