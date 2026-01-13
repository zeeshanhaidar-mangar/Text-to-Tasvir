import streamlit as st
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from io import BytesIO

# Set page configuration
st.set_page_config(
    page_title="Text to Image Generator",
    page_icon="🎨",
    layout="centered"
)

@st.cache_resource
def load_pipeline():
    """
    Load the Stable Diffusion pipeline.
    This function is cached to avoid reloading the model on every interaction.
    """
    try:
        model_name = "segmind/SSD-1B"
        
        # Detect device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        # Load pipeline
        pipe = StableDiffusionPipeline.from_pretrained(
            model_name,
            torch_dtype=dtype,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        
        # Enable memory optimizations
        if device == "cpu":
            pipe.enable_attention_slicing()
        
        return pipe, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def generate_single_image(pipe, prompt, negative_prompt, num_steps, guidance_scale, seed=None):
    """
    Generate a single image from text prompt.
    
    Args:
        pipe: Diffusion pipeline
        prompt: Text prompt
        negative_prompt: Negative prompt
        num_steps: Number of inference steps
        guidance_scale: Guidance scale
        seed: Random seed (optional)
    
    Returns:
        PIL.Image: Generated image or None if error
    """
    try:
        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator(device=pipe.device).manual_seed(seed)
        else:
            generator = None
        
        # Generate image
        result = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt else None,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        return result.images[0]
    except Exception as e:
        st.error(f"Error generating image: {str(e)}")
        return None

def convert_to_grayscale(image):
    """
    Convert RGB image to grayscale.
    
    Args:
        image: PIL Image
    
    Returns:
        PIL.Image: Grayscale image
    """
    return image.convert("L")

def create_animated_gif(images, frame_duration):
    """
    Create an animated GIF from a list of images.
    
    Args:
        images: List of PIL Images
        frame_duration: Duration of each frame in milliseconds
    
    Returns:
        BytesIO: GIF data
    """
    try:
        gif_buffer = BytesIO()
        
        # Save as animated GIF
        images[0].save(
            gif_buffer,
            format="GIF",
            save_all=True,
            append_images=images[1:],
            duration=frame_duration,
            loop=0
        )
        
        gif_buffer.seek(0)
        return gif_buffer
    except Exception as e:
        st.error(f"Error creating GIF: {str(e)}")
        return None

def image_to_bytes(image, format="PNG"):
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image
        format: Image format (PNG, JPEG, etc.)
    
    Returns:
        BytesIO: Image data
    """
    img_buffer = BytesIO()
    image.save(img_buffer, format=format)
    img_buffer.seek(0)
    return img_buffer

def main():
    st.title("🎨 Text to Image Generator")
    st.write("Generate images from text descriptions using AI!")
    
    # Warning about first run
    st.info("ℹ️ First run will download the model (~2-3 GB). Generation speed depends on your hardware.")
    
    # Load pipeline
    with st.spinner("Loading AI model..."):
        pipe, device = load_pipeline()
    
    if pipe is None:
        st.error("Failed to load model. Please refresh the page.")
        return
    
    st.success(f"✅ Model loaded successfully! Running on: {device.upper()}")
    
    st.write("---")
    
    # Input fields
    col1, col2 = st.columns([2, 1])
    
    with col1:
        prompt = st.text_area(
            "Prompt",
            value="A futuristic city at sunset, cyberpunk style",
            height=100,
            help="Describe the image you want to generate"
        )
        
        negative_prompt = st.text_area(
            "Negative Prompt (Optional)",
            value="",
            height=80,
            help="Describe what you don't want in the image"
        )
    
    with col2:
        image_type = st.selectbox(
            "Image Type",
            ["Color (RGB) Image", "Grayscale Image", "Animated GIF"],
            help="Choose the type of output"
        )
        
        num_steps = st.slider(
            "Inference Steps",
            min_value=4,
            max_value=20,
            value=8,
            help="More steps = better quality but slower"
        )
        
        guidance_scale = st.slider(
            "Guidance Scale",
            min_value=1.0,
            max_value=12.0,
            value=7.0,
            step=0.5,
            help="Higher values follow prompt more closely"
        )
    
    # Additional settings for animated GIF
    num_frames = None
    frame_duration = None
    
    if image_type == "Animated GIF":
        col3, col4 = st.columns(2)
        with col3:
            num_frames = st.slider(
                "Number of Frames",
                min_value=5,
                max_value=20,
                value=10,
                help="Number of frames in the GIF"
            )
        with col4:
            frame_duration = st.slider(
                "Frame Duration (ms)",
                min_value=100,
                max_value=500,
                value=200,
                step=50,
                help="Duration of each frame in milliseconds"
            )
    
    st.write("---")
    
    # Generate button
    if st.button("🎨 Generate", type="primary", use_container_width=True):
        if not prompt.strip():
            st.warning("⚠️ Please enter a prompt to generate an image.")
            return
        
        # Color (RGB) Image
        if image_type == "Color (RGB) Image":
            with st.spinner("Generating image..."):
                image = generate_single_image(
                    pipe,
                    prompt,
                    negative_prompt,
                    num_steps,
                    guidance_scale
                )
                
                if image:
                    st.success("Image generated successfully!")
                    st.image(image, caption="Generated Image", use_container_width=True)
                    
                    # Download button
                    img_bytes = image_to_bytes(image, format="PNG")
                    st.download_button(
                        label="⬇️ Download Image",
                        data=img_bytes,
                        file_name="generated_image.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        # Grayscale Image
        elif image_type == "Grayscale Image":
            with st.spinner("Generating image..."):
                image = generate_single_image(
                    pipe,
                    prompt,
                    negative_prompt,
                    num_steps,
                    guidance_scale
                )
                
                if image:
                    # Convert to grayscale
                    gray_image = convert_to_grayscale(image)
                    
                    st.success("Grayscale image generated successfully!")
                    st.image(gray_image, caption="Generated Grayscale Image", use_container_width=True)
                    
                    # Download button
                    img_bytes = image_to_bytes(gray_image, format="PNG")
                    st.download_button(
                        label="⬇️ Download Image",
                        data=img_bytes,
                        file_name="generated_grayscale.png",
                        mime="image/png",
                        use_container_width=True
                    )
        
        # Animated GIF
        elif image_type == "Animated GIF":
            with st.spinner(f"Generating {num_frames} frames for GIF..."):
                images = []
                progress_bar = st.progress(0)
                
                # Generate multiple frames with different seeds
                for i in range(num_frames):
                    seed = 42 + i  # Use sequential seeds for variation
                    image = generate_single_image(
                        pipe,
                        prompt,
                        negative_prompt,
                        num_steps,
                        guidance_scale,
                        seed=seed
                    )
                    
                    if image:
                        images.append(image)
                        progress_bar.progress((i + 1) / num_frames)
                
                progress_bar.empty()
                
                if len(images) == num_frames:
                    # Create GIF
                    with st.spinner("Creating animated GIF..."):
                        gif_bytes = create_animated_gif(images, frame_duration)
                        
                        if gif_bytes:
                            st.success("Animated GIF generated successfully!")
                            st.image(gif_bytes, caption="Generated Animated GIF", use_container_width=True)
                            
                            # Download button
                            gif_bytes.seek(0)
                            st.download_button(
                                label="⬇️ Download GIF",
                                data=gif_bytes,
                                file_name="generated_animation.gif",
                                mime="image/gif",
                                use_container_width=True
                            )
                else:
                    st.error("Failed to generate all frames for the GIF.")
    
    # Footer
    st.write("---")
    st.caption("Powered by Hugging Face Diffusers | Model: segmind/SSD-1B")
    
    # Tips
    with st.expander("💡 Tips for Better Results"):
        st.write("""
        - **Be specific**: Detailed prompts produce better results
        - **Use style keywords**: Add terms like "photorealistic", "oil painting", "3D render"
        - **Negative prompts**: Help exclude unwanted elements (e.g., "blurry, low quality")
        - **Inference steps**: 8-12 steps usually give good results; more steps take longer
        - **Guidance scale**: 7-9 works well for most prompts; higher values follow prompt more strictly
        - **Animated GIFs**: Use 8-12 frames for smooth animations without long wait times
        """)

if __name__ == "__main__":
    main()
