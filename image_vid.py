import os
import cv2
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from insightface.app import FaceAnalysis
import shutil
# Define directories
new_images_dir = input("Enter a new value for images_dir(the def is C:\python\images ): ")
new_vid_dir = input("Enter a new value for vid_dir(the def is "+r"C:\python\vids ):")
images_dir = new_images_dir if new_images_dir else r"C:\python\images"
vid_dir = new_vid_dir if new_vid_dir else r"C:\python\vids"
existingdir=True
if(not os.path.exists(images_dir) or not os.path.exists(vid_dir)):
    existingdir=False


 

def extract_embedding(embedder, face_data):
    try:
        if face_data and 'embedding' in face_data:
            embedding = face_data['embedding']
            return embedding
        else:
            print("No faces detected.")  # Debug log
            return None
    except Exception as e:
        print("Error during embedding extraction:", e)  # Debug log
        return None

def load_model_for_embedding():
    try:
        model_name = 'arcface_r100_v1'
        embedder = FaceAnalysis(model=model_name)
        embedder.prepare(ctx_id=0, det_thresh=0.3, det_size=(64, 64))
        return embedder
    except Exception as e:
        print("Error during embedder model initialization:", e)
        return None

def create_output_directory(image_file,path):
    image_name = os.path.splitext(image_file)[0]
    if(path!=images_dir+"\\archive"):
        output_dir = os.path.join(path, image_name)
    else:
        output_dir=path
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    return output_dir


def process_images(images_dir):
    embedder = load_model_for_embedding()
    embeddings = []

    if embedder is None:
        return
    #subfolder_path, _,

    for image_files in os.listdir(images_dir):
        if os.path.isfile(os.path.join(images_dir, image_files)):
            image_path = os.path.join(images_dir, image_files)
            img = cv2.imread(image_path)
            faces = embedder.get(img)

            if faces:
                for face in faces:
                    embedding = extract_embedding(embedder, face)
                    if embedding is not None:
                        embeddings.append((embedding, image_files))

    return embeddings
def compare_embeddings_button(embeddings, video_path, embedder):
    results = []
    for embedding, image_file in embeddings:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        found_match = False
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            faces = embedder.get(frame)
            if faces:
                for face in faces:
                    video_embedding = extract_embedding(embedder, face)

                    if video_embedding is not None:
                        similarity = cosine_similarity([embedding], [video_embedding])[0][0]
                        threshold = 0.5

                        if similarity >= threshold:
                            results.append(
                                f"Match found in {os.path.basename(video_path)} for image: {image_file} with similarity: {similarity}")
                            found_match = True
                            break

                if found_match:
                    break

        cap.release()

    return results

# Function to compare embeddings
# ... [rest of the imports and function definitions remain unchanged]

def compare_embeddings(embeddings):
    for video_file in os.listdir(vid_dir):
        video_path = os.path.join(vid_dir, video_file)
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Failed to open video file: {video_path}")
            continue

        processed_images = set()  # To track which images have been processed

        while True:
            ret, frame = cap.read()
            if not ret:
                break  # End of the video

            faces = embedder.get(frame)
            if not faces:
                continue

            for face in faces:
                video_embedding = extract_embedding(embedder, face)
                if video_embedding is None:
                    continue

                found_match = False
                for embedding, image_file in embeddings:
                    if image_file in processed_images:
                        continue

                    # Calc
                    # ulate cosine similarity between the embeddings
                    similarity = cosine_similarity([embedding], [video_embedding])[0][0]

                    # Set a threshold for similarity (adjust as needed)
                    threshold = 0.47

                    if similarity >= threshold:
                        print(f"Match found in {video_file} for image: {image_file} with similarity: {similarity}")
                        output_dir = create_output_directory(image_file,r"C:\python")
                        
                        matched_frame_path = os.path.join(output_dir, f"frame_{video_file}.jpg")
                        cv2.imwrite(matched_frame_path, frame)
                        
                        matched_image_path = os.path.join(output_dir, image_file)
                        image_path = os.path.join(images_dir, image_file)
                        if(os.path.exists(image_path)):
                         shutil.copy(image_path, matched_image_path)
                        
                        matched_video_path = os.path.join(output_dir, video_file)
                        shutil.copy(video_path, matched_video_path)


                        output_dir = create_output_directory(image_file,images_dir+r"\archive")
                        matched_video_path = os.path.join(output_dir, image_file)
                        image_path = os.path.join(images_dir, image_file)
                        if(os.path.exists(image_path)):
                         shutil.move(image_path, matched_video_path)




                        

                        processed_images.add(image_file)  # Mark this image as processed
                        found_match = True
                        break

                if found_match:
                    break

        cap.release()

if __name__ == "__main__":
    # Load the embedding model
    if(existingdir==True):
    
     embedder = load_model_for_embedding()

     if embedder:
        # embedding of images to pass to compare_embeddings
        embeddings = process_images(images_dir)  #  function to get embeddings

        if embeddings:
            compare_embeddings(embeddings)
            message = "Video processing complete."
        else:
            message = "No embeddings found."
     else:
        message = "Failed to load the embedding model."
    else:
        message = "the path doesn't exist,try again"

        

    print(message)
