from ultralytics import YOLO
import cv2
import os
from pathlib import Path
from tqdm import tqdm

class CattleROIExtractor:
    def __init__(self, yolo_model_path='yolov8n.pt'):
        self.model = YOLO(yolo_model_path)
        self.cattle_classes = [19]  # COCO class for cow
        
    def extract_roi(self, image_path, output_dir, confidence_threshold=0.4):
        """
        Extract cattle ROI from image
        Returns: List of cropped ROI images
        """
        
        # Run detection
        results = self.model.predict(
            image_path, 
            classes=self.cattle_classes,
            conf=confidence_threshold,
            verbose=False
        )
        
        img = cv2.imread(str(image_path))
        rois = []
        
        for r in results:
            boxes = r.boxes
            for idx, box in enumerate(boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                
                # Add padding (10%)
                h, w = y2 - y1, x2 - x1
                padding_h, padding_w = int(h * 0.1), int(w * 0.1)
                
                y1 = max(0, y1 - padding_h)
                y2 = min(img.shape[0], y2 + padding_h)
                x1 = max(0, x1 - padding_w)
                x2 = min(img.shape[1], x2 + padding_w)
                
                # Crop ROI
                roi = img[y1:y2, x1:x2]
                
                if roi.size > 0:
                    rois.append(roi)
        
        return rois
    
    def process_dataset(self, input_dir, output_dir):
        """
        Process entire dataset and extract ROIs
        """
        os.makedirs(output_dir, exist_ok=True)
        
        image_files = list(Path(input_dir).rglob('*.jpg')) + \
                     list(Path(input_dir).rglob('*.jpeg')) + \
                     list(Path(input_dir).rglob('*.png'))
        
        print(f"Processing {len(image_files)} images...")
        
        successful = 0
        failed = 0
        
        for img_path in tqdm(image_files):
            try:
                rois = self.extract_roi(str(img_path), output_dir)
                
                if rois:
                    # Save first ROI (usually the main animal)
                    output_path = os.path.join(
                        output_dir,
                        img_path.name
                    )
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    cv2.imwrite(output_path, rois[0])
                    successful += 1
                else:
                    failed += 1
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                failed += 1
        
        print(f"\nROI Extraction Complete!")
        print(f"Successful: {successful}, Failed: {failed}")

# Usage
if __name__ == "__main__":
    extractor = CattleROIExtractor()
    
    # Process each split
    for split in ['train', 'val', 'test']:
        print(f"\n=== Processing {split} split ===")
        extractor.process_dataset(
            input_dir=f'data/processed/{split}/images',
            output_dir=f'data/processed/{split}/roi_images'
        )
