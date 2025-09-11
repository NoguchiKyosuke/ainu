#!/usr/bin/env python3
"""
Download all audio files from Asai Take folktale collection
https://www.aa.tufs.ac.jp/~mmine/kiki_gen/murasaki/asai01.html
"""

import requests
from bs4 import BeautifulSoup
import re
from urllib.parse import urljoin
import os
import time
from pathlib import Path
import json
from tqdm import tqdm

def download_asai_collection():
    # Create directories for downloaded audio
    base_dir = Path('data/samples')
    asai_dir = base_dir / 'asai_take_stories'
    asai_dir.mkdir(exist_ok=True)
    
    # Create metadata file to track downloads
    metadata_file = asai_dir / 'metadata.json'
    metadata = {
        'collection': 'Asai Take Folktale Collection',
        'source': 'https://www.aa.tufs.ac.jp/~mmine/kiki_gen/murasaki/asai01.html',
        'stories': {}
    }
    
    # Base URL for the story collection
    base_url = 'https://www.aa.tufs.ac.jp/~mmine/kiki_gen/murasaki/'
    
    print('Starting download of complete Asai Take audio collection...')
    print(f'Target directory: {asai_dir}')
    
    # Generate all story page URLs (At01 to At54)
    story_urls = []
    for i in range(1, 55):  # 1 to 54
        story_url = f'{base_url}at{i:02d}aj.html'
        story_urls.append((i, story_url))
    
    print(f'Found {len(story_urls)} story pages to process')
    
    total_downloaded = 0
    total_audio_files = 0
    
    # Process all stories with progress bar
    for story_num, story_url in tqdm(story_urls, desc="Processing stories"):
        story_id = f"At{story_num:02d}"
        print(f'\\nProcessing {story_id}: {story_url}')
        
        try:
            response = requests.get(story_url, timeout=30)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Look for audio links in this page
            audio_links = []
            for link in soup.find_all('a', href=True):
                href = link['href']
                if any(ext in href.lower() for ext in ['.wav', '.mp3', '.au', '.aiff']):
                    full_url = urljoin(story_url, href)
                    audio_links.append(full_url)
            
            print(f'  Found {len(audio_links)} audio files')
            total_audio_files += len(audio_links)
            
            # Download each audio file with progress bar
            story_downloaded = 0
            for i, audio_url in enumerate(tqdm(audio_links, desc=f"  {story_id} audio", leave=False)):
                filename = f'{story_id}_{i+1:03d}.wav'
                filepath = asai_dir / filename
                
                # Skip if already exists
                if filepath.exists():
                    print(f'    ⚡ Skipping (exists): {filename}')
                    story_downloaded += 1
                    continue
                
                try:
                    audio_response = requests.get(audio_url, timeout=30)
                    audio_response.raise_for_status()
                    
                    with open(filepath, 'wb') as f:
                        f.write(audio_response.content)
                    
                    story_downloaded += 1
                    print(f'    ✓ Downloaded: {filename} ({len(audio_response.content)} bytes)')
                    
                except Exception as e:
                    print(f'    ✗ Failed to download {audio_url}: {e}')
                
                time.sleep(0.3)  # Be respectful to the server
            
            total_downloaded += story_downloaded
            
            # Update metadata
            metadata['stories'][story_id] = {
                'url': story_url,
                'audio_files_found': len(audio_links),
                'audio_files_downloaded': story_downloaded,
                'title': f'Story {story_num}'  # Could be enhanced with actual title
            }
            
            print(f'  ✓ {story_id}: {story_downloaded}/{len(audio_links)} files downloaded')
            
        except Exception as e:
            print(f'  ✗ Error processing {story_url}: {e}')
            metadata['stories'][story_id] = {
                'url': story_url,
                'error': str(e)
            }
        
        # Save metadata periodically
        if story_num % 5 == 0:
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    # Final metadata save
    metadata['total_stories'] = len(story_urls)
    metadata['total_audio_files_found'] = total_audio_files
    metadata['total_audio_files_downloaded'] = total_downloaded
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f'\\n🎉 Download completed!')
    print(f'📊 Statistics:')
    print(f'   Stories processed: {len(story_urls)}')
    print(f'   Total audio files found: {total_audio_files}')
    print(f'   Successfully downloaded: {total_downloaded}')
    print(f'   Download directory: {asai_dir}')
    print(f'   Metadata saved to: {metadata_file}')
    
    return asai_dir, total_downloaded, total_audio_files

if __name__ == '__main__':
    download_asai_collection()
