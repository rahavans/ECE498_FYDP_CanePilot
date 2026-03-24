"""Main app for Hailo Whisper"""


### YOU NEED TO ADD
### /home/canepilot/Projects/Hailo-Application-Code-Examples/runtime/hailo-8/python/speech_recognition
### TO YOUR PYTHONPATH

import os
import threading
import time
import tempfile
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import select
import sys
import queue
from common.preprocessing import preprocess, improve_input_audio
from common.postprocessing import clean_transcription
from common.audio_utils import load_audio
from hailo_platform import (VDevice, HailoSchedulingAlgorithm, HEF, ConfigureParams, 
                           HailoStreamInterface, InputVStreamParams, OutputVStreamParams, 
                           InferVStreams, FormatType)
from transformers import AutoTokenizer
import numpy as np
from common.postprocessing import apply_repetition_penalty

# Whisper expects 16kHz mono audio
SAMPLE_RATE = 16000
CHANNELS = 1

def record_audio_continuous(duration, audio_path, stop_event):
    """
    Record audio from the microphone with ability to stop early via stop_event.
    
    Args:
        duration (int): Maximum duration of the recording in seconds.
        audio_path (str): Path to save the recorded audio.
        stop_event (threading.Event): Event to signal early stopping.

    Returns:
        np.ndarray: Recorded audio data.
    """
    q = queue.Queue()
    recorded_frames = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)
        q.put(indata.copy())

    print(f"Recording for up to {duration} seconds. Call finish() to stop...")

    start_time = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype="float32",
                        callback=audio_callback):
        while True:
            if time.time() - start_time >= duration:
                print("Max duration reached.")
                break
            if stop_event.is_set():
                print("Recording stopped.")
                break
            try:
                frame = q.get(timeout=0.1)
                recorded_frames.append(frame)
            except queue.Empty:
                continue

    print("Recording finished. Processing...")

    if not recorded_frames:
        return np.array([])

    audio_data = np.concatenate(recorded_frames, axis=0)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    wav.write(audio_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    return audio_data


class ListeningSession:
    """
    A wrapper class that manages non-blocking audio recording and blocking inference using streaming pipeline.
    """
    
    def __init__(self, max_duration=60, variant="base"):
        self.max_duration = max_duration
        self.variant = variant
        self.recording_thread = None
        self.is_recording = False
        self.recorded_audio = None
        self.audio_path = None
        self.start_time = None
        self.stop_event = threading.Event()
        self.recording_error = None
        
    def _record_audio_thread(self):
        """Record audio in a separate thread"""
        try:
            # Create a temporary file for this recording session
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                self.audio_path = temp_file.name
            
            # Record audio using the continuous recording function
            self.recorded_audio = record_audio_continuous(
                self.max_duration, 
                self.audio_path, 
                self.stop_event
            )
            
        except Exception as e:
            print(f"Recording error: {e}")
            self.recording_error = e
        finally:
            self.is_recording = False
    
    def finish(self):
        """
        Stop recording and perform inference using streaming pipeline. This is a blocking operation.
        
        Returns:
            str: The transcribed text from the recorded audio
        """
        if not self.is_recording:
            return ""
            
        # Signal the recording thread to stop
        self.stop_event.set()
        
        # Wait for the recording thread to finish
        if self.recording_thread and self.recording_thread.is_alive():
            self.recording_thread.join(timeout=10.0)  # Wait up to 10 seconds
        
        # Check for recording errors
        if self.recording_error:
            print(f"Recording failed: {self.recording_error}")
            return ""
            
        if self.recorded_audio is None or len(self.recorded_audio) == 0:
            print("No audio recorded.")
            return ""
        
        try:
            # Process the recorded audio using streaming pipeline
            sampled_audio, start_time = improve_input_audio(self.recorded_audio, vad=True)
            chunk_offset = start_time - 0.2 if start_time else 0
            if chunk_offset < 0:
                chunk_offset = 0

            # Set chunk length based on variant
            chunk_length = 5 if self.variant == "base" else 10

            mel_spectrograms = preprocess(
                sampled_audio,
                is_nhwc=True,
                chunk_length=chunk_length,
                chunk_offset=chunk_offset
            )

            transcription = ""
            for mel in mel_spectrograms:
                result = run_inference(mel, variant=self.variant)
                transcription = clean_transcription(result)
            
            return transcription
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Inference error: {e}")
            return ""
        finally:
            # Clean up temporary audio file
            if self.audio_path and os.path.exists(self.audio_path):
                try:
                    os.unlink(self.audio_path)
                except:
                    pass


def start_listening(max_duration=60, variant="base"):
    """
    Start non-blocking audio recording and return a ListeningSession wrapper.
    
    Args:
        max_duration (int): Maximum recording duration in seconds (default: 60)
        variant (str): Whisper model variant ("base" or "tiny")
        
    Returns:
        ListeningSession: A wrapper object with a finish() method
        
    Raises:
        RuntimeError: If the streaming pipeline is not initialized
    """
    if streaming_encoder_network_group is None:
        raise RuntimeError("Streaming pipeline not initialized. Call init_whisper_pipeline() first.")
    
    session = ListeningSession(max_duration, variant)
    session.is_recording = True
    session.recording_thread = threading.Thread(target=session._record_audio_thread)
    session.recording_thread.daemon = True  # Allow program to exit even if thread is running
    session.recording_thread.start()
    
    return session


# Original record_audio function for backward compatibility
def record_audio(duration, audio_path):
    """
    Record audio from the microphone and save it as a WAV file. The user has the possibility to stop the recording earlier by pressing Enter on the keyboard.

    Args:
        duration (int): Duration of the recording in seconds.

    Returns:
        np.ndarray: Recorded audio data.
    """
    q = queue.Queue()
    recorded_frames = []

    def audio_callback(indata, frames, time_info, status):
        if status:
            print("Status:", status)
        q.put(indata.copy())

    print(f"Recording for up to {duration} seconds. Press Enter to stop early...")

    start_time = time.time()
    with sd.InputStream(samplerate=SAMPLE_RATE,
                        channels=CHANNELS,
                        dtype="float32",
                        callback=audio_callback):
        # Set stdin to non-blocking line-buffered mode
        sys.stdin = open('/dev/stdin')
        while True:
            if time.time() - start_time >= duration:
                print("Max duration reached.")
                break
            try:
                # Check if Enter was pressed (non-blocking)
                if select.select([sys.stdin], [], [], 0.0)[0]:
                    sys.stdin.read(1)  # consume the newline
                    print("Early stop requested.")
                    break
            except:
                pass  # Handle any stdin errors gracefully
            
            try:
                frame = q.get(timeout=0.1)
                recorded_frames.append(frame)
            except queue.Empty:
                continue

    print("Recording finished. Processing...")

    if not recorded_frames:
        return np.array([])

    audio_data = np.concatenate(recorded_frames, axis=0)
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=1)

    wav.write(audio_path, SAMPLE_RATE, (audio_data * 32767).astype(np.int16))
    return audio_data


audio_path = "sampled_audio.wav"
is_nhwc = True

def init(vdevice=None, variant="base"):
    """
    Initialize the Whisper pipeline using the streaming method.
    
    Args:
        vdevice: Pre-initialized VDevice to use for inference. If None, will create a new one.
        variant: Whisper model variant ("base" or "tiny")
    """
    print(f"Initializing Whisper {variant} pipeline...")
    init_whisper_pipeline(vdevice=vdevice, variant=variant)

async def listen(time_duration=10, variant="base"):  # Renamed parameter to avoid conflict with time module
    """
    Listen and transcribe using the streaming pipeline.
    
    Args:
        time_duration: Recording duration in seconds
        variant: Whisper model variant
    
    Returns:
        str: Transcribed text
    """
    return await listen_streaming(time_duration=time_duration, variant=variant)

def stop():
    """
    Stop and clean up the streaming Whisper pipeline.
    """
    stop_whisper_pipeline()


# Global variables for streaming pipeline
streaming_encoder_network_group = None
streaming_decoder_network_group = None
streaming_encoder_inp_vstreams = None
streaming_encoder_out_vstreams = None
streaming_decoder_inp_vstreams = None
streaming_decoder_out_vstreams = None
streaming_tokenizer = None
streaming_token_embedding_weight = None
streaming_onnx_add_input = None
streaming_vdevice = None
decoder_network_group_params = None
encoder_network_group_params = None

ASSERT_BASE_PATH = "/home/canepilot/Projects/Hailo-Application-Code-Examples/runtime/hailo-8/python/speech_recognition/app"

def _load_streaming_tokenizer_assets(variant="base"):
    """Load tokenizer and tokenization assets for streaming pipeline."""
    global streaming_tokenizer, streaming_token_embedding_weight, streaming_onnx_add_input
    
    # Load tokenizer
    streaming_tokenizer = AutoTokenizer.from_pretrained(f"openai/whisper-{variant}")
    
    # Load token embedding assets
    import os
    
    token_embedding_path = os.path.join(ASSERT_BASE_PATH, f"decoder_assets/{variant}/decoder_tokenization/token_embedding_weight_{variant}.npy")
    onnx_add_input_path = os.path.join(ASSERT_BASE_PATH, f"decoder_assets/{variant}/decoder_tokenization/onnx_add_input_{variant}.npy")
    
    streaming_token_embedding_weight = np.load(token_embedding_path)
    streaming_onnx_add_input = np.load(onnx_add_input_path)

def _streaming_tokenization(decoder_input_ids):
    """Perform tokenization for streaming pipeline."""
    global streaming_token_embedding_weight, streaming_onnx_add_input
    
    # Embedding lookup
    gather_output = streaming_token_embedding_weight[decoder_input_ids]
    # Add bias
    add_output = gather_output + streaming_onnx_add_input
    # Insert dimension at axis=1
    unsqueeze_output = np.expand_dims(add_output, axis=1)
    # Transpose to NHWC format
    transpose_output = np.transpose(unsqueeze_output, (0, 2, 1, 3))
    
    return transpose_output

encoder_path = "/home/canepilot/Projects/Hailo-Application-Code-Examples/runtime/hailo-8/python/speech_recognition/app/hefs/h8/base/base-whisper-encoder-5s.hef"
decoder_path = "/home/canepilot/Projects/Hailo-Application-Code-Examples/runtime/hailo-8/python/speech_recognition/app/hefs/h8/base/base-whisper-decoder-fixed-sequence-matmul-split.hef"

def init_whisper_pipeline(vdevice=None, variant="base"):
    """
    Initialize Whisper pipeline using virtual streams and network groups.
    
    Args:
        vdevice: Pre-initialized VDevice to use. If None, will create a new one.
        variant: Whisper model variant ("base" or "tiny")
    
    Returns:
        vdevice: The VDevice being used (for cleanup later)
    """
    global streaming_encoder_network_group, streaming_decoder_network_group
    global streaming_encoder_inp_vstreams, streaming_encoder_out_vstreams
    global streaming_decoder_inp_vstreams, streaming_decoder_out_vstreams
    global streaming_vdevice
    global encoder_network_group_params, decoder_network_group_params
    
    print(f"Initializing Whisper {variant} pipeline with streaming method...")
    
    # Create or use provided VDevice
    if vdevice is None:
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        streaming_vdevice = VDevice(params)
        print("Created new VDevice for streaming pipeline")
    else:
        streaming_vdevice = vdevice
        print("Using provided VDevice for streaming pipeline")
    
    # Create HEF objects
    encoder_hef = HEF(encoder_path)
    decoder_hef = HEF(decoder_path)
    
    # Create configure parameters
    encoder_cfg = ConfigureParams.create_from_hef(encoder_hef, interface=HailoStreamInterface.PCIe)
    decoder_cfg = ConfigureParams.create_from_hef(decoder_hef, interface=HailoStreamInterface.PCIe)
    
    # Configure vdevice to get network groups
    encoder_network_groups = streaming_vdevice.configure(encoder_hef, encoder_cfg)
    decoder_network_groups = streaming_vdevice.configure(decoder_hef, decoder_cfg)
    
    streaming_encoder_network_group = encoder_network_groups[0]
    streaming_decoder_network_group = decoder_network_groups[0]
    
    # Create vstream parameters
    streaming_encoder_inp_vstreams = InputVStreamParams.make_from_network_group(
        streaming_encoder_network_group, quantized=False, format_type=FormatType.FLOAT32)
    streaming_encoder_out_vstreams = OutputVStreamParams.make_from_network_group(
        streaming_encoder_network_group, quantized=False, format_type=FormatType.FLOAT32)
    
    streaming_decoder_inp_vstreams = InputVStreamParams.make_from_network_group(
        streaming_decoder_network_group, quantized=False, format_type=FormatType.FLOAT32)
    streaming_decoder_out_vstreams = OutputVStreamParams.make_from_network_group(
        streaming_decoder_network_group, quantized=False, format_type=FormatType.FLOAT32)
    
    # Load tokenizer and assets
    _load_streaming_tokenizer_assets(variant)

    # Create network group parameters
    encoder_network_group_params = streaming_encoder_network_group.create_params()
    decoder_network_group_params = streaming_decoder_network_group.create_params()

    print("✓ Streaming pipeline initialized successfully")
    return streaming_vdevice


def run_inference(input_mel, variant="base"):
    """
    Run inference using the streaming pipeline with network group activation.
    Treats network.activate() as a lock - only one network active at a time.
    
    Args:
        input_mel: Preprocessed mel spectrogram input
        variant: Whisper model variant ("base" or "tiny")
    
    Returns:
        str: Transcribed text
    """
    global streaming_encoder_network_group, streaming_decoder_network_group
    global streaming_encoder_inp_vstreams, streaming_encoder_out_vstreams
    global streaming_decoder_inp_vstreams, streaming_decoder_out_vstreams
    global streaming_tokenizer
    
    if streaming_encoder_network_group is None:
        raise RuntimeError("Streaming pipeline not initialized. Call init_whisper_pipeline() first.")
    
    # Set decoding sequence length based on variant
    decoding_sequence_length = 32 if variant == "tiny" else 24
    
    # Get sorted output names for concatenation
    decoder_hef = HEF(decoder_path)
    sorted_output_names = decoder_hef.get_sorted_output_names()
    
    # Step 1: Activate encoder network and run encoder inference
    print("Activating encoder network...")
    with streaming_encoder_network_group.activate(encoder_network_group_params):
        with InferVStreams(streaming_encoder_network_group, streaming_encoder_inp_vstreams, streaming_encoder_out_vstreams) as encoder_infer_pipeline:
            
            # Prepare encoder input
            input_mel = np.ascontiguousarray(input_mel)
            encoder_input_data = {list(streaming_encoder_inp_vstreams.keys())[0]: input_mel}
            
            # Run encoder inference
            print("Running encoder inference...")
            encoder_outputs = encoder_infer_pipeline.infer(encoder_input_data)
            encoded_features = list(encoder_outputs.values())[0]
            print("Encoder inference complete, deactivating encoder network...")
    
    # Initialize decoder state
    start_token_id = [50258]
    decoder_input_ids = np.array(
        [[start_token_id[0]]], dtype=np.int64
    )  # Shape (1,1)
    decoder_input_ids = np.concatenate(
        [decoder_input_ids, np.zeros((1, decoding_sequence_length - 1), dtype=np.int64)], axis=1
    )
    
    generated_tokens = []
    # Step 2: Run decoder iteratively - activate/deactivate for each iteration
    # Activate decoder network for this iteration only
    with streaming_decoder_network_group.activate(decoder_network_group_params):
        with InferVStreams(streaming_decoder_network_group, streaming_decoder_inp_vstreams, streaming_decoder_out_vstreams) as decoder_infer_pipeline:
            for i in range(decoding_sequence_length - 1):
                print(f"Decoder iteration {i+1}/{decoding_sequence_length-1}: Activating decoder network...")
        
                # Tokenize current sequence
                tokenized_ids = _streaming_tokenization(decoder_input_ids)
                
                # Prepare decoder input data
                decoder_input_keys = list(streaming_decoder_inp_vstreams.keys())
                decoder_input_data = {
                    decoder_input_keys[0]: encoded_features,  # input_layer1
                    decoder_input_keys[1]: tokenized_ids      # input_layer2
                }
                
                # Run decoder inference
                print(f"Running decoder inference for iteration {i+1}...")
                decoder_outputs_dict = decoder_infer_pipeline.infer(decoder_input_data)
                
                # Handle decoder outputs
                output_arrays = [decoder_outputs_dict[name].squeeze(0) for name in sorted_output_names]
                
                # Concatenate decoder outputs along axis 2
                decoder_outputs = np.concatenate(output_arrays, axis=2)
                
                # Apply repetition penalty and get next token
                repetition_penalty = 1.5
                logits = apply_repetition_penalty(decoder_outputs[:, i], generated_tokens, penalty=repetition_penalty)
                next_token = np.argmax(logits)
                
                generated_tokens.append(next_token)
                decoder_input_ids[0][i + 1] = next_token
                print(f"Generated token {i+1}: {next_token}, deactivating decoder network...")
                
                # Check for end of sequence
                if next_token == streaming_tokenizer.eos_token_id:
                    print("End of sequence token detected, stopping generation...")
                    break
    
    # Convert tokens to text
    transcription = streaming_tokenizer.decode(generated_tokens, skip_special_tokens=True)
    print(f"Final transcription: '{transcription}'")
    return transcription

def stop_whisper_pipeline():
    """
    Stop and clean up the streaming Whisper pipeline.
    """
    global streaming_encoder_network_group, streaming_decoder_network_group
    global streaming_encoder_inp_vstreams, streaming_encoder_out_vstreams
    global streaming_decoder_inp_vstreams, streaming_decoder_out_vstreams
    global streaming_tokenizer, streaming_token_embedding_weight
    global streaming_onnx_add_input, streaming_vdevice
    
    print("Stopping streaming Whisper pipeline...")
    
    # Reset all global variables
    streaming_encoder_network_group = None
    streaming_decoder_network_group = None
    streaming_encoder_inp_vstreams = None
    streaming_encoder_out_vstreams = None
    streaming_decoder_inp_vstreams = None
    streaming_decoder_out_vstreams = None
    streaming_tokenizer = None
    streaming_token_embedding_weight = None
    streaming_onnx_add_input = None
    
    # Release VDevice if we created it
    if streaming_vdevice is not None:
        try:
            streaming_vdevice.release()
            print("✓ VDevice released")
        except Exception as e:
            print(f"Warning: Error releasing VDevice: {e}")
        streaming_vdevice = None
    
    print("✓ Streaming pipeline stopped and cleaned up")

async def listen_streaming(time_duration=10, variant="base"):
    """
    Listen and transcribe using the streaming pipeline.
    
    Args:
        time_duration: Recording duration in seconds
        variant: Whisper model variant
    
    Returns:
        str: Transcribed text
    """
    # Record audio
    sampled_audio = record_audio(time_duration, audio_path=audio_path)
    
    # Process audio
    sampled_audio = load_audio(audio_path)
    sampled_audio, start_time = improve_input_audio(sampled_audio, vad=True)
    chunk_offset = start_time - 0.2 if start_time else 0
    if chunk_offset < 0:
        chunk_offset = 0
    
    # Set chunk length based on variant
    chunk_length = 5 if variant == "base" else 10
    
    mel_spectrograms = preprocess(
        sampled_audio,
        is_nhwc=is_nhwc,
        chunk_length=chunk_length,
        chunk_offset=chunk_offset
    )
    
    # Run inference on each mel spectrogram
    transcription = ""
    for mel in mel_spectrograms:
        result = run_inference(mel, variant=variant)
        transcription = clean_transcription(result)
    
    return transcription


if __name__ == "__main__":
    import asyncio
    vdevice = None
    
    def test_async_wrapper():
        """Test the new async wrapper functionality"""
        print("\n=== Testing Async Wrapper (start_listening/finish) ===")
        print("This will start recording in the background...")
        
        try:
            # Start non-blocking recording
            session = start_listening(max_duration=30, variant="base")
            print("✓ Recording started in background thread")
            
            # Simulate doing other work while recording continues
            print("Doing other work while recording continues...")
            import time
            for i in range(5):
                print(f"  Working... {i+1}/5")
                time.sleep(1)
            
            print("Stopping recording and processing audio...")
            # Stop recording and get transcription (blocking)
            transcription = session.finish()
            
            if transcription:
                print(f"✓ Transcription result: '{transcription}'")
            else:
                print("✗ No transcription received (empty result)")
                
        except Exception as e:
            print(f"✗ Error during async wrapper test: {e}")

    def test_streaming_pipeline():
        """Test the new streaming pipeline functionality"""
        print("\n=== Testing Streaming Pipeline ===")
        print("This will initialize the streaming pipeline and record audio...")
        
        try:
            # Record audio
            print("Recording for 10 seconds (press Enter to stop early)...")
            transcription = asyncio.run(listen_streaming(time_duration=10, variant="base"))
            
            if transcription:
                print(f"✓ Streaming transcription result: '{transcription}'")
            else:
                print("✗ No transcription received (empty result)")
                
        except Exception as e:
            print(f"✗ Error during streaming pipeline test: {e}")

    async def test_original_function():
        """Test the original listen() function"""
        print("\n=== Testing Original listen() Function ===")
        print("This will record for 10 seconds (press Enter to stop early)...")
        
        try:
            transcription = await listen(time_duration=10, variant="base")
            
            if transcription:
                print(f"✓ Transcription result: '{transcription}'")
            else:
                print("✗ No transcription received (empty result)")
                
        except Exception as e:
            print(f"✗ Error during original function test: {e}")
    
    def main():
        """Main test function"""
        print("Hailo Whisper Test Script")
        print("=" * 50)
        
        # Choose initialization method
        print("Select initialization method:")
        print("1. Use shared VDevice (recommended for multiple pipelines)")
        print("2. Use internal VDevice (simpler, pipeline manages its own)")
        
        while True:
            choice = input("Enter your choice (1-2): ").strip()
            if choice in ["1", "2"]:
                break
            print("Invalid choice. Please enter 1 or 2.")
        
        use_shared_vdevice = (choice == "1")
        if use_shared_vdevice:
            params = VDevice.create_params()
            params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
            vdevice = VDevice(params)
        else:
            vdevice = None
        
        # Initialize the pipeline
        print("Initializing Hailo Whisper pipeline...")
        try:
            init(vdevice=vdevice, variant="base")
            print("✓ Pipeline initialized successfully")
        except Exception as e:
            print(f"✗ Failed to initialize pipeline: {e}")
            print("Make sure all HEF files are available and paths are correct.")
            if use_shared_vdevice and vdevice:
                try:
                    vdevice.release()
                except:
                    pass
            return
        
        # Test menu
        while True:
            print("\nSelect test to run:")
            print("1. Test async wrapper (start_listening/finish)")
            print("2. Test original function (listen)")
            print("3. Test streaming pipeline (new method)")
            print("4. Run all tests")
            print("5. Exit")
            
            choice = input("Enter your choice (1-5): ").strip()
            
            if choice == "1":
                test_async_wrapper()
            elif choice == "2":
                asyncio.run(test_original_function())
            elif choice == "3":
                test_streaming_pipeline()
            elif choice == "4":
                test_async_wrapper()
                asyncio.run(test_original_function())
                test_streaming_pipeline()
            elif choice == "5":
                print("Exiting...")
                break
            else:
                print("Invalid choice. Please enter 1-5.")
        
        # Cleanup
        print("\nCleaning up...")
        try:
            stop()
            print("✓ Pipeline stopped successfully")
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")
        
        # Clean up shared VDevice if we created one
        if use_shared_vdevice and vdevice:
            try:
                print("Cleaning up shared VDevice...")
                vdevice.release()
                print("✓ Shared VDevice released")
            except Exception as e:
                print(f"Warning: Error releasing shared VDevice: {e}")
    
    # Run the main test function
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        try:
            stop()
        except:
            pass
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        try:
            stop()
        except:
            pass