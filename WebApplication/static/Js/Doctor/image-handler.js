// Image handling module
const ImageHandler = {
  uploadedImageUrls: [],
  currentFrameIndex: 0,
  imageAnnotations: [],
  isPlaying: false,
  playbackInterval: null,

  handleImageUpload: (event) => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.multiple = true;
    fileInput.accept = "image/*";
    fileInput.webkitdirectory = true;
    fileInput.directory = true;

    fileInput.addEventListener("change", (e) => {
      const files = Array.from(e.target.files)
        .filter(file => file.type.startsWith("image/"))
        .sort((a, b) => {
          const nameA = a.name.toLowerCase();
          const nameB = b.name.toLowerCase();
          return nameA.localeCompare(nameB, undefined, { numeric: true });
        });

      if (files.length === 0) {
        alert("No image files found in the selected folder.");
        return;
      }

      ImageHandler.clearImages();
      ImageHandler.processFiles(files);
    });

    fileInput.click();
  },

  processFiles: (files) => {
    const thumbnailGrid = document.getElementById("thumbnail-grid");
    thumbnailGrid.innerHTML = "";
    
    files.forEach((file, idx) => {
      const reader = new FileReader();
      reader.onload = function(e) {
        ImageHandler.uploadedImageUrls.push(e.target.result);
        ImageHandler.createThumbnail(e.target.result, idx);
        
        if (idx === 0) {
          ImageHandler.updatePreview(0);
        }
      };
      reader.readAsDataURL(file);
    });

    document.querySelector(".image-upload-section").classList.add("has-files");
    ImageHandler.enableControls();
  },

  createThumbnail: (imageUrl, index) => {
    const thumbnailGrid = document.getElementById("thumbnail-grid");
    const img = document.createElement("img");
    img.src = imageUrl;
    img.classList.add("thumbnail-img");
    img.dataset.index = index;
    img.addEventListener("click", () => ImageHandler.updatePreview(index));
    thumbnailGrid.appendChild(img);
  },

  updatePreview: (index) => {
    index = parseInt(index, 10);
    if (index < 0 || index >= ImageHandler.uploadedImageUrls.length) {
      return;
    }

    ImageHandler.currentFrameIndex = index;
    
    const frameCounter = document.getElementById("frame-counter");
    if (frameCounter) {
      frameCounter.textContent = `Frame ${index + 1} of ${ImageHandler.uploadedImageUrls.length}`;
    }

    const previewImage = document.getElementById("preview-image");
    if (previewImage) {
      previewImage.src = ImageHandler.uploadedImageUrls[index];
      previewImage.classList.add("loaded");
    }

    ImageHandler.updateThumbnails(index);
    ImageHandler.updateControls();
  },

  updateThumbnails: (activeIndex) => {
    document.querySelectorAll(".thumbnail-img").forEach(thumb => {
      if (parseInt(thumb.dataset.index) === activeIndex) {
        thumb.classList.add("active");
      } else {
        thumb.classList.remove("active");
      }
    });
  },

  clearImages: () => {
    ImageHandler.uploadedImageUrls = [];
    ImageHandler.currentFrameIndex = 0;
    ImageHandler.imageAnnotations = [];
    const thumbnailGrid = document.getElementById("thumbnail-grid");
    if (thumbnailGrid) {
      thumbnailGrid.innerHTML = "";
    }
    const previewImage = document.getElementById("preview-image");
    if (previewImage) {
      previewImage.src = "https://placehold.co/300x200/f0f0f0/cccccc?text=Preview";
      previewImage.classList.remove("loaded");
    }
    document.querySelector(".image-upload-section").classList.remove("has-files");
  },

  enableControls: () => {
    const controls = [
      document.getElementById("annotation-phase-select"),
      document.getElementById("play-pause-btn"),
      document.getElementById("prev-frame-btn"),
      document.getElementById("next-frame-btn")
    ];
    
    controls.forEach(control => {
      if (control) {
        control.disabled = false;
      }
    });
  },

  updateControls: () => {
    const playPauseBtn = document.getElementById("play-pause-btn");
    const prevFrameBtn = document.getElementById("prev-frame-btn");
    const nextFrameBtn = document.getElementById("next-frame-btn");
    
    if (prevFrameBtn) {
      prevFrameBtn.disabled = ImageHandler.currentFrameIndex === 0;
    }
    if (nextFrameBtn) {
      nextFrameBtn.disabled = ImageHandler.currentFrameIndex >= ImageHandler.uploadedImageUrls.length - 1;
    }
    if (playPauseBtn) {
      playPauseBtn.innerHTML = ImageHandler.isPlaying ? 
        '<i class="fas fa-pause"></i>' : 
        '<i class="fas fa-play"></i>';
    }
  },

  togglePlayback: () => {
    if (!ImageHandler.isPlaying) {
      ImageHandler.startPlayback();
    } else {
      ImageHandler.stopPlayback();
    }
  },

  startPlayback: () => {
    ImageHandler.isPlaying = true;
    const playPauseBtn = document.getElementById("play-pause-btn");
    if (playPauseBtn) {
      playPauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
    }
    
    ImageHandler.playbackInterval = setInterval(() => {
      if (ImageHandler.currentFrameIndex < ImageHandler.uploadedImageUrls.length - 1) {
        ImageHandler.updatePreview(ImageHandler.currentFrameIndex + 1);
      } else {
        ImageHandler.stopPlayback();
      }
    }, 500);
  },

  stopPlayback: () => {
    ImageHandler.isPlaying = false;
    const playPauseBtn = document.getElementById("play-pause-btn");
    if (playPauseBtn) {
      playPauseBtn.innerHTML = '<i class="fas fa-play"></i>';
    }
    
    if (ImageHandler.playbackInterval) {
      clearInterval(ImageHandler.playbackInterval);
      ImageHandler.playbackInterval = null;
    }
  },

  goToPreviousFrame: () => {
    if (ImageHandler.currentFrameIndex > 0) {
      ImageHandler.updatePreview(ImageHandler.currentFrameIndex - 1);
    }
  },

  goToNextFrame: () => {
    if (ImageHandler.currentFrameIndex < ImageHandler.uploadedImageUrls.length - 1) {
      ImageHandler.updatePreview(ImageHandler.currentFrameIndex + 1);
    }
  }
};

// Export the module
window.ImageHandler = ImageHandler;