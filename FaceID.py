#Import Kivy dependencies First
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

#Kivy UX components
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label

#Kivy other stuff
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

#Import other dependencies
import cv2
import tensorflow as tf
from layers import L1Dist
import os
import numpy as np

#Build app and layout
class CamApp(App):
    def build (self):
        #Main Layout components
        self.web_cam = Image(size_hint=(1,.8))
        self.button = Button(text='Verify',on_press=self.verify, size_hint=(1,.1))
        self.verification_label = Label(text='Verification Uninitiated', size_hint=(1,.1))

        #Add items to layout
        layout = BoxLayout(orientation = 'vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        #Load TensorFlow Keras Model
        self.model = tf.keras.models.load_model('siamesemodelv2.h5', custom_objects={'L1Dist':L1Dist})

        
        #Setup Video Capture Device
        self.capture = cv2.VideoCapture(1)
        Clock.schedule_interval(self.update, 1.0/33.0)
        
        return layout
    
    # Run Continously to get Webcam feed
    def update(self,*args):
        
        #Read Frame from OpenCV
        ret, frame =self.capture.read()
        frame = frame[250:550, 450:650+250, :]

        #Flip horizontal and convert image to texture
        #Render Webcam
        buf = cv2.flip(frame,0).tostring()
        img_texture = Texture.create(size=(frame.shape[1],frame.shape[0]),colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr',bufferfmt= 'ubyte')
        self.web_cam.texture = img_texture

    #Load image from file and convert to 100x100 px
    def preprocess(self, file_path):
        try:
            byte_img = tf.io.read_file(file_path)
            img = tf.io.decode_jpeg(byte_img)
            img = tf.image.resize(img, (100,100))
            img = img / 255.0
            return img
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None



    #Verification function to verify person
    def verify(self, *args):

        # Define the is_image_file function
        def is_image_file(filename):
            valid_image_extensions = [".jpg", ".jpeg", ".png", ".gif", ".bmp"]
            return any(filename.lower().endswith(ext) for ext in valid_image_extensions)

        #Specify Thresholds
        detection_threshold= 0.5
        verification_threshold = 0.5

        #Capture input image from Webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        frame = frame[250:550, 450:650+250, :]
        cv2.imwrite(SAVE_PATH,frame)

        results = []
        verification_images_path = os.path.join('application_data', 'verification_images')
        positive_detections = 0  # Initialize a counter for positive detections

        for image in os.listdir(verification_images_path):
            if not is_image_file(image):
                continue  # Skip this file if it's not an image
            
            input_img_path = os.path.join('application_data', 'input_image', 'input_image.jpg')
            validation_img_path = os.path.join(verification_images_path, image)
            
            input_img = self.preprocess(input_img_path)
            validation_img = self.preprocess(validation_img_path)
            
            # If preprocess returns None (in case of an error), skip this iteration
            if input_img is None or validation_img is None:
                print(f"Skipping image due to preprocessing error: {validation_img_path}")
                continue
            
            # Make Predictions 
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)
            
            # Detection Threshold: Metric above which a prediction is considered positive
            if result > detection_threshold:
                positive_detections += 1
            
        
        # Verification Threshold: Proportion of positive predictions / total positive samples 
        total_images = len([image for image in os.listdir(verification_images_path) if is_image_file(image)])
        verification = positive_detections / total_images
        verified = verification > verification_threshold

        #Set verification text
        self.verification_label.text = 'Verified' if verified == True else 'Unverified'
        

        #Log out Details
        Logger.info(results)
        Logger.info(np.sum(np.array(results)>0.2))
        Logger.info(np.sum(np.array(results)>0.4))
        Logger.info(np.sum(np.array(results)>0.5))
        Logger.info(np.sum(np.array(results)>0.8))
        Logger.info(results)
        Logger.info(positive_detections)
        Logger.info(verification)
        Logger.info(verified)

        return results, verified

     

    
if __name__=='__main__':
    CamApp().run()