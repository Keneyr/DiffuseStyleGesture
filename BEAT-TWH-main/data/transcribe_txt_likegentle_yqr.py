import whisperx
import json
import sys
import os

def transcribe_audio(audio_file, text_file):
    """Transcribe audio to text file"""
    model = whisperx.load_model("base", device="cpu", compute_type="float32")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio)
    transcript = " ".join([segment["text"] for segment in result["segments"]])
    
    with open(text_file, 'w', encoding='utf-8') as f:
        f.write(transcript)

def align_text(audio_file, text_file, output_file):
    """Align text with audio and create alignment file"""
    with open(text_file, 'r', encoding='utf-8') as f:
        transcript = f.read().strip()
    
    model = whisperx.load_model("base", device="cpu", compute_type="float32")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio)
    
    model_a, metadata = whisperx.load_align_model(language_code="en", device="cpu")
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device="cpu")
    
    words = []
    char_offset = 0
    
    for segment in aligned_result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                word_text = word_info["word"].strip()
                
                # Skip words without timing information
                if "start" not in word_info or "end" not in word_info:
                    continue
                    
                start_offset = transcript.lower().find(word_text.lower(), char_offset)
                if start_offset == -1:
                    start_offset = char_offset
                end_offset = start_offset + len(word_text)
                char_offset = end_offset
                
                words.append({
                    "alignedWord": word_text.lower(),
                    "case": "success",
                    "end": word_info["end"],
                    "endOffset": end_offset,
                    "start": word_info["start"],
                    "startOffset": start_offset,
                    "word": word_text
                })
    
    output = {
        "transcript": transcript,
        "words": words
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

def audio_to_aligned(audio_file, output_file):
    """Directly convert audio to aligned format (transcribe + align in one step)"""
    model = whisperx.load_model("base", device="cpu", compute_type="float32")
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio)
    
    # Get transcript
    transcript = " ".join([segment["text"] for segment in result["segments"]])
    
    # Align
    model_a, metadata = whisperx.load_align_model(language_code="en", device="cpu")
    aligned_result = whisperx.align(result["segments"], model_a, metadata, audio, device="cpu")
    
    words = []
    char_offset = 0
    
    for segment in aligned_result["segments"]:
        if "words" in segment:
            for word_info in segment["words"]:
                word_text = word_info["word"].strip()
                
                # Skip words without timing information
                if "start" not in word_info or "end" not in word_info:
                    continue
                    
                start_offset = transcript.lower().find(word_text.lower(), char_offset)
                if start_offset == -1:
                    start_offset = char_offset
                end_offset = start_offset + len(word_text)
                char_offset = end_offset
                
                words.append({
                    "alignedWord": word_text.lower(),
                    "case": "success",
                    "end": word_info["end"],
                    "endOffset": end_offset,
                    "start": word_info["start"],
                    "startOffset": start_offset,
                    "word": word_text
                })
    
    output = {
        "transcript": transcript,
        "words": words
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    if len(sys.argv) == 3:
        audio_file, output_file = sys.argv[1], sys.argv[2]
        
        # Check if second argument ends with .txt or .json to determine mode
        if output_file.endswith('.txt'):
            # Mode 1: transcribe audio to text
            transcribe_audio(audio_file, output_file)
        else:
            # Mode 3: direct audio to aligned format
            audio_to_aligned(audio_file, output_file)
            
    elif len(sys.argv) == 4:
        # Mode 2: align text with audio
        audio_file, text_file, output_file = sys.argv[1], sys.argv[2], sys.argv[3]
        align_text(audio_file, text_file, output_file)
    else:
        print("Usage:")
        print("  Transcribe: python3 transcribe_txt_likegentle.py <audio_file> <text_file>")
        print("  Align: python3 transcribe_txt_likegentle.py <audio_file> <text_file> <output_file>")
        print("  Direct: python3 transcribe_txt_likegentle.py <audio_file> <aligned_output_file>")