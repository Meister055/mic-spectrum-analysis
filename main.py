from PIL import Image, ImageDraw, ImageFont
import numpy as np
from scipy import signal
import os

def process_spectrum(input_path, output_path, title, num_sectors=32):
    """
    Process a spectrum image by:
    1. Detecting the left and right edges of the spectrum
    2. Dividing the spectrum into evenly spaced sectors
    3. Calculating average color for each sector
    4. Annotating the image with all information
    
    Args:
        input_path: Path to input image
        output_path: Path to save processed image
        title: Title to display on image
        num_sectors: Number of sectors to divide spectrum into
    """

    # Add column number font
    try:
        numFont = ImageFont.truetype("arial.ttf", 12)
    except:
        numFont = ImageFont.load_default()

    try:
        # Load image
        img = Image.open(input_path)
        if img.mode == 'P':
            img = img.convert('RGB')
        img_array = np.array(img)
        height, width = img_array.shape[:2]
        draw = ImageDraw.Draw(img)

        # Add title
        try:
            font = ImageFont.truetype("arial.ttf", 40)
        except:
            font = ImageFont.load_default()
        title_bbox = draw.textbbox((0, 0), title, font=font)
        title_x = (width - (title_bbox[2] - title_bbox[0])) // 2
        draw.rectangle([title_x-5, 5, title_x+title_bbox[2]+5, title_bbox[3]+15], fill=(255,255,255))
        draw.text((title_x, 10), title, fill=(0,0,0), font=font)

        # Detect spectrum edges - MODIFIED SECTION
        avg_colors = np.mean(img_array, axis=0)
        grayscale = np.mean(avg_colors, axis=1)
        grayscale = (grayscale - np.min(grayscale)) / (np.max(grayscale) - np.min(grayscale) + 1e-10)
        gradient = np.gradient(grayscale)
        gradient_magnitude = np.abs(gradient)
    
        # Ignore outer 10% of right side for edge detection
        ignore_right = int(width * 0.9)  # Only look for right edge in first 90% of image
        modified_gradient = gradient_magnitude.copy()
        modified_gradient[ignore_right:] = 0  # Zero out gradient magnitude in right 10%
    
        # Find peaks in the modified gradient
        peaks, _ = signal.find_peaks(modified_gradient, height=0.1, distance=20)
        
        if len(peaks) >= 2:
            strongest_peaks = peaks[np.argsort(modified_gradient[peaks])[-2:]]
            left_edge = min(strongest_peaks)
            right_edge = max(strongest_peaks)
            
            # Ensure right edge isn't in the ignored 10% region
            right_edge = min(right_edge, ignore_right)
        else:
            left_edge = width // 4
            right_edge = min(3 * width // 4, ignore_right)  # Also limit default right edge

        # Draw spectrum boundaries
        draw.rectangle([left_edge, 0, right_edge, height], outline=(0,255,0), width=2)
        draw.line([(left_edge,0), (left_edge,height)], fill=(0,0,0), width=3)
        draw.line([(right_edge,0), (right_edge,height)], fill=(0,0,0), width=3)
        
        # Create 32 sectors between edges
        sector_width = (right_edge - left_edge) / num_sectors
        sector_boundaries = [int(left_edge + i*sector_width) for i in range(num_sectors+1)]
        sector_colors = []

        # Draw sector lines and calculate average colors
        for i in range(num_sectors+1):
            x = sector_boundaries[i]
            if 0 < i < num_sectors:
                draw.line([(x,0), (x,height)], fill=(0,0,0), width=1)
            
            if i < num_sectors:
                # Calculate average sector color
                sector = img_array[:, sector_boundaries[i]:sector_boundaries[i+1]]
                avg_color = np.mean(sector, axis=(0,1)).astype(int)
                sector_colors.append(avg_color)

            # Annotate column number
            if i < num_sectors:
                draw.text((x + 5, height - 60), str(i+1), fill=(0,0,0), font=numFont)
                

        # Save results
        img.save(output_path)
        print(f"Processed image saved to {output_path}")
        
        # Return analysis data
        return {
            'edges': (left_edge, right_edge),
            'sector_boundaries': sector_boundaries,
            'sector_colors': sector_colors,
            'spectrum_width': right_edge - left_edge
        }

    except Exception as e:
        print(f"Error processing image: {e}")
        return None
    


def process_folder(input_folder, output_folder, num_sectors=32):
    """
    Process all images in a folder
    
    Args:
        input_folder: Path to folder containing input images
        output_folder: Path to save processed images
        num_sectors: Number of sectors to divide spectrum into
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all image files in input folder
    image_files = [f for f in os.listdir(input_folder) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif'))]
    
    all_results = {}
    
    for image_file in image_files:
        try:
            # Construct paths
            input_path = os.path.join(input_folder, image_file)
            output_path = os.path.join(output_folder, f"{image_file}")
            
            # Use filename (without extension) as title
            title = os.path.splitext(image_file)[0]
            
            print(f"\nProcessing {image_file}...")
            result = process_spectrum(
                input_path=input_path,
                output_path=output_path,
                title=title,
                num_sectors=num_sectors
            )
            
            if result:
                all_results[image_file] = result
                
                # Save color data for this image
                color_file = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_processed.csv")
                with open(color_file, "w") as f:
                    f.write(f"Sector #, Opinion\n")
                    for i, color in enumerate(result['sector_colors']):
                        hsv = tuple(int(c) for c in np.round(np.array(color) / 255 * 360).astype(int))
                        if hsv[0] >= 200:
                            sell = "Don't sell"
                        elif hsv[0] >= 125:
                            sell = "Squeeze room"
                        elif hsv[0] < 125:
                            sell = "Sell"
                        f.write(f"Sector {i+1}, {sell}\n")
                
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return all_results

if __name__ == "__main__":
    print("Processing spectrum images...")
    
    input_folder = input("Enter input folder path: ").strip()
    output_folder = input("Enter output folder path: ").strip()
    
    results = process_folder(
        input_folder=input_folder,
        output_folder=output_folder,
        num_sectors=32
    )
    
    print("\nProcessing complete.")
    if results:
        print(f"\nProcessed {len(results)} images:")
        for filename, data in results.items():
            print(f"- {filename}: width={data['spectrum_width']}px")