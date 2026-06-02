#!/usr/bin/env python3
"""
Script to embed photos from Desktop/val folder into the Valentine's game HTML
"""
import os
import base64
from pathlib import Path

def get_desktop_path():
    """Get the desktop path"""
    # Path to val folder in workspace
    desktop = Path(__file__).parent / "val"
    return desktop

def embed_photos_in_html():
    """Embed photos into the HTML file"""
    
    # Get photos from Desktop/val folder
    desktop_val = get_desktop_path()
    
    if not desktop_val.exists():
        print(f"❌ Folder not found: {desktop_val}")
        print(f"Please make sure you have a 'val' folder on your Desktop with your photos!")
        return
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.webp']
    photos = []
    
    for file in desktop_val.iterdir():
        if file.suffix.lower() in image_extensions:
            photos.append(file)
    
    if not photos:
        print(f"❌ No photos found in {desktop_val}")
        print(f"Please add some photos (.jpg, .png, etc.) to the val folder!")
        return
    
    # Take first 5 photos
    photos = photos[:5]
    print(f"✅ Found {len(photos)} photos!")
    
    # Convert photos to base64
    photo_data = []
    for photo in photos:
        print(f"   📸 Processing: {photo.name}")
        with open(photo, 'rb') as f:
            encoded = base64.b64encode(f.read()).decode('utf-8')
            # Determine mime type
            ext = photo.suffix.lower()
            if ext in ['.jpg', '.jpeg']:
                mime = 'image/jpeg'
            elif ext == '.png':
                mime = 'image/png'
            elif ext == '.gif':
                mime = 'image/gif'
            elif ext == '.webp':
                mime = 'image/webp'
            else:
                mime = 'image/jpeg'
            
            photo_data.append(f'data:{mime};base64,{encoded}')
    
    # Read the current HTML
    html_path = Path(__file__).parent / 'valentine_game.html'
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    
    # Create the photos array as JavaScript
    photos_js = ',\n                '.join([f'"{data}"' for data in photo_data])
    
    # Replace the photos array initialization
    html_content = html_content.replace(
        'let photos = [];',
        f'''let photos = [
                {photos_js}
            ];'''
    )
    
    # Update the start screen to hide upload section and auto-enable button
    html_content = html_content.replace(
        '<button class="start-btn" id="startBtn" disabled>',
        '<button class="start-btn" id="startBtn">'
    )
    
    # Hide the upload area
    html_content = html_content.replace(
        '.upload-area {',
        '.upload-area {\n            display: none;'
    )
    
    # Write the new HTML file
    output_path = Path(__file__).parent / 'valentine_game_ready.html'
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"\n✅ SUCCESS! Your Valentine's game is ready!")
    print(f"📁 File created: {output_path}")
    print(f"\n💕 Next steps:")
    print(f"   1. Open valentine_game_ready.html in a browser to test")
    print(f"   2. Send this file to your girlfriend, or")
    print(f"   3. Upload it to Netlify/GitHub Pages to get a link")
    print(f"\n🔥 Good luck, Romeo! 😘")

if __name__ == '__main__':
    embed_photos_in_html()
