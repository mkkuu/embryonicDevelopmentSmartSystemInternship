/**
 * ========================================
 * EMBRYO ANALYSIS SYSTEM - DOCTOR INTERFACE
 * ========================================
 * 
 * This JavaScript module provides the frontend functionality for the Doctor's
 * Embryo Management interface in the Embryo Analysis Web Application.
 * 
 * Features:
 * - Embryo record CRUD operations
 * - Image upload and management with drag-and-drop
 * - Frame preview and thumbnail navigation
 * - Phase annotation system
 * - AI-powered developmental transition prediction
 * - Real-time data validation
 * - Responsive table management with pagination
 * - File management and CSV export
 * 
 * Key Components:
 * - Form Management: Add, update, delete embryo records
 * - Image Uploader: Drag-and-drop image handling with preview
 * - Frame Navigation: Thumbnail-based frame selection and preview
 * - Phase Annotation: Dropdown-based phase selection for each frame
 * - AI Prediction: Integration with backend prediction models
 * - Table Management: Pagination, search, and data display
 * 
 * API Integration:
 * - RESTful API calls to Flask backend
 * - File upload handling with FormData
 * - Real-time data synchronization
 * - Error handling and user feedback
 * 
 * UI/UX Features:
 * - Modal dialogs for confirmations and feedback
 * - Loading states and progress indicators
 * - Responsive design for different screen sizes
 * - Keyboard navigation support
 * - Drag-and-drop file upload
 * 
 * Author: LSL Team
 * Version: 1.0
 * Last Updated: 2025-10-04
 */

document.addEventListener("DOMContentLoaded", function () {
  // --- MAIN FORM FIELDS ---
  const Parent_ID = document.getElementById("Parent_ID"),
    Record_Date = document.getElementById("Record_Date"),
    Parent_Contact = document.getElementById("Parent_Contact");

  const inputsToVerify = [Record_Date, Parent_Contact];

  // Set today's date as default
  const today = new Date();
  const todayString = today.toISOString().split("T")[0];
  Record_Date.value = todayString;

  // --- MAIN ACTION BUTTONS ---
  const clearButton = document.getElementById("Clear"),
    addButton = document.getElementById("Add"),
    updateButton = document.getElementById("Update"),
    deleteButton = document.getElementById("Delete");

  // --- TABLE & PAGINATION ELEMENTS ---
  const tableHeader = document.querySelector(".data-table thead"),
    tableBody = document.querySelector(".data-table tbody"),
    searchInput = document.getElementById("search"),
    noDataRow = document.querySelector(".no-data-row");
  const entriesPerPageSelect =
      document.getElementById("entries-per-page"),
    prevButton = document.getElementById("prev-button"),
    nextButton = document.getElementById("next-button"),
    entriesInfo = document.getElementById("entries-info");

  // --- SORTING VARIABLES ---
  let sortColumn = null;
  let sortDirection = null;

  // --- FRAME PREVIEW ELEMENTS ---
  const uploadBox = document.getElementById("upload-box");
  const previewSection = document.getElementById("preview-section");
  const currentImage = document.getElementById("current-image");
  const frameText = document.getElementById("frame-text");
  const phaseSelector = document.getElementById("phase-selector");
  const thumbnailStrip = document.getElementById("thumbnail-strip");
  const prevBtn = document.getElementById("prev-btn");
  const playBtn = document.getElementById("play-btn");
  const nextBtn = document.getElementById("next-btn");
  const blastocysteGrade1 = document.getElementById("blastocyste-grade1");
  const blastocysteGrade2 = document.getElementById("blastocyste-grade2");
  const blastocysteGrade3 = document.getElementById("blastocyste-grade3");
  const pgtaSelect = document.getElementById("pgta-select");
  const liveBirthSelect = document.getElementById("live-birth-select");
  const predictButton = document.getElementById("predict-button");
  const predictParameter = document.getElementById("predict-parameter");

  // --- MODAL ELEMENTS ---
  const modal = document.getElementById("modal");
  const modalTitle = document.getElementById("modal-title");
  const modalMessage = document.getElementById("modal-message");
  const modalButtons = document.getElementById("modal-buttons");

  // --- STATE VARIABLES ---
  let allDataRows = [],
    currentPage = 1,
    rowsPerPage = parseInt(entriesPerPageSelect.value, 10),
    sortDirections = {};
  let frameImages = [],
    frameUrls = [],
    frameAnnotations = [],
    currentFrameIndex = 0,
    isPlaying = false,
    playbackInterval = null,
    nextParentId = 1,
    currentRecordId = null;

  const annotationPhases = [
    "N/A",
    "tPNa",
    "tPNf",
    "t2",
    "t3",
    "t4",
    "t5",
    "t6",
    "t7",
    "t8",
    "tM",
    "tSB",
    "tB",
    "tEB",
    "tHB",
    "Anomaly",
  ];

  // --- SORTING LOGIC ---
  function naturalSort(a, b) {
    const re = /(\d+)/g;
    const aParts = a.name.split(re);
    const bParts = b.name.split(re);

    for (let i = 0; i < Math.min(aParts.length, bParts.length); i++) {
      const aPart = aParts[i];
      const bPart = bParts[i];
      if (i % 2) {
        const aNum = parseInt(aPart, 10);
        const bNum = parseInt(bPart, 10);
        if (aNum !== bNum) return aNum - bNum;
      } else {
        if (aPart !== bPart) return aPart.localeCompare(bPart);
      }
    }
    return aParts.length - bParts.length;
  }

  function sortTable(column) {
    if (sortColumn === column) {
      if (sortDirection === "asc") {
        sortDirection = "desc";
      } else if (sortDirection === "desc") {
        sortDirection = null;
        sortColumn = null;
      } else {
        sortDirection = "asc";
      }
    } else {
      sortColumn = column;
      sortDirection = "asc";
    }

    if (sortDirection) {
      allDataRows.sort((rowA, rowB) => {
        let valA, valB;

        if (column === "parentId") {
          valA = parseInt(rowA.dataset.parentId, 10);
          valB = parseInt(rowB.dataset.parentId, 10);
        } else if (column === "grading") {
          const gradesA =
            [
              rowA.dataset.studentBlastocyste_g1,
              rowA.dataset.studentBlastocyste_g2,
              rowA.dataset.studentBlastocyste_g3,
            ]
              .filter((grade) => grade && grade !== "")
              .join("/") || "-";
          const gradesB =
            [
              rowB.dataset.studentBlastocyste_g1,
              rowB.dataset.studentBlastocyste_g2,
              rowB.dataset.studentBlastocyste_g3,
            ]
              .filter((grade) => grade && grade !== "")
              .join("/") || "-";
          valA = gradesA;
          valB = gradesB;
        } else if (column === "aneuploidy") {
          valA = rowA.dataset.studentAneuploidy;
          valB = rowB.dataset.studentAneuploidy;
        } else if (column === "live_birth") {
          valA = rowA.dataset.studentLive_birth;
          valB = rowB.dataset.studentLive_birth;
        } else {
          valA = rowA.dataset[column];
          valB = rowB.dataset[column];
        }

        let comparison = 0;
        if (valA < valB) comparison = -1;
        else if (valA > valB) comparison = 1;

        return sortDirection === "desc" ? -comparison : comparison;
      });
    } else {
      allDataRows.sort((rowA, rowB) => {
        const idA = parseInt(rowA.dataset.parentId, 10);
        const idB = parseInt(rowB.dataset.parentId, 10);
        return idA - idB;
      });
    }

    updateSortIcons();
    displayPage();
  }

  function updateSortIcons() {
    tableHeader.querySelectorAll(".sort-icons i").forEach((icon) => {
      icon.classList.remove("active");
    });
    if (sortColumn) {
      const currentTh = tableHeader.querySelector(
        `th[data-column="${sortColumn}"]`
      );
      if (currentTh) {
        if (sortDirection === "asc") {
          currentTh.querySelector(".fa-sort-up").classList.add("active");
        } else if (sortDirection === "desc") {
          currentTh
            .querySelector(".fa-sort-down")
            .classList.add("active");
        }
      }
    }
  }

  // --- PAGINATION & DISPLAY LOGIC ---
  function displayPage() {
    const filteredRows = getFilteredRows();
    tableBody.innerHTML = "";
    tableBody.appendChild(noDataRow);

    const start = (currentPage - 1) * rowsPerPage;
    const end = start + rowsPerPage;
    filteredRows
      .slice(start, end)
      .forEach((row) => tableBody.insertBefore(row, noDataRow));

    updatePaginationControls(filteredRows);
    updateEntriesInfo(filteredRows);
    toggleNoDataRow(filteredRows.length);
  }
  function getFilteredRows() {
    return allDataRows.filter((row) =>
      row.textContent
        .toLowerCase()
        .includes(searchInput.value.toLowerCase())
    );
  }
  function toggleNoDataRow(count) {
    noDataRow.style.display = count === 0 ? "table-row" : "none";
  }
  function updateEntriesInfo(rows) {
    const total = rows.length,
      start = total > 0 ? (currentPage - 1) * rowsPerPage + 1 : 0,
      end = Math.min(start + rowsPerPage - 1, total);
    entriesInfo.textContent = `Showing ${start} to ${end} of ${total} entries`;
  }
  function updatePaginationControls(rows) {
    const totalPages = Math.ceil(rows.length / rowsPerPage);
    prevButton.disabled = currentPage === 1;
    nextButton.disabled = currentPage === totalPages || totalPages === 0;
  }

  // --- INPUT VALIDATION ---
  function verifyInputs() {
    resetInputErrors();
    let isValid = true;

    // Check if images are uploaded
    if (frameImages.length === 0) {
      showModal("Validation Error", "Please upload embryo images before adding a record.");
        isValid = false;
    }

    // Check required grading fields
    if (blastocysteGrade1.value === "") {
      blastocysteGrade1.classList.add("input-error");
      isValid = false;
    }
    if (blastocysteGrade2.value === "") {
      blastocysteGrade2.classList.add("input-error");
      isValid = false;
    }
    if (blastocysteGrade3.value === "") {
      blastocysteGrade3.classList.add("input-error");
      isValid = false;
    }

    // PGTA and Live Birth are optional - no validation needed

    if (!isValid) {
      showModal("Validation Error", "Please fill in all required fields (Images and Grading fields are required).");
    }

    return isValid;
  }
  function resetInputErrors() {
    inputsToVerify.forEach((input) =>
      input.classList.remove("input-error")
    );
    blastocysteGrade1.classList.remove("input-error");
    blastocysteGrade2.classList.remove("input-error");
    blastocysteGrade3.classList.remove("input-error");
    pgtaSelect.classList.remove("input-error");
    liveBirthSelect.classList.remove("input-error");
  }

  // --- CRUD & FORM LOGIC ---
  function updateNextId() {
    const maxId = allDataRows.reduce(
      (max, row) => Math.max(max, parseInt(row.dataset.parentId, 10)),
      0
    );
    nextParentId = maxId + 1;
  }

  function clearFields() {
    // Reset the entire form
    document.querySelector(".parent-form").reset();
    
    // Set today's date
    const today = new Date();
    const todayString = today.toISOString().split("T")[0];
    Record_Date.value = todayString;
    Record_Date.readOnly = true;
    
    // Clear ID field - always keep it read-only
    Parent_ID.value = "";
    Parent_ID.readOnly = true;
    
    // Clear all select dropdowns
    blastocysteGrade1.selectedIndex = 0;
    blastocysteGrade2.selectedIndex = 0;
    blastocysteGrade3.selectedIndex = 0;
    pgtaSelect.selectedIndex = 0;
    liveBirthSelect.selectedIndex = 0;
    
    // Clear all frames and images
    clearFrames();
    
    // Remove any row selection styling
    removeRowStyling();
    
    // Clear any input error styling
    resetInputErrors();
    
    // Reset current record tracking
    currentRecordId = null;
    
    // Clear search input
    searchInput.value = "";
    
    // Reset pagination to first page
    currentPage = 1;
    
    // Reset sorting
    sortColumn = null;
    sortDirection = null;
    updateSortIcons();
    
    // Refresh the table display
    displayPage();
  }

  function removeRowStyling() {
    const prev = document.querySelector(".data-table tbody .selected");
    if (prev) prev.classList.remove("selected");
  }
  function highlightRow(row) {
    removeRowStyling();
    row.classList.add("selected");
  }

  async function populateFields(row) {
    resetInputErrors();
    const d = row.dataset;
    Parent_ID.value = d.parentId;
    Parent_ID.readOnly = true; // Make ID read-only when populated from row selection
    Record_Date.value = d.studentDate;
    Record_Date.type = "date";
    Parent_Contact.value = d.parentContact;

    blastocysteGrade1.value = d.studentBlastocyste_g1 || "";
    blastocysteGrade2.value = d.studentBlastocyste_g2 || "";
    blastocysteGrade3.value = d.studentBlastocyste_g3 || "";
    pgtaSelect.value = d.studentAneuploidy || "";
    liveBirthSelect.value = d.studentLive_birth || "";
    
    currentRecordId = d.parentId;
    
    // Load images and annotations for this embryo
    await loadEmbryoImagesAndAnnotations(d.parentId);
  }

  function addRowClickListener(row) {
    row.addEventListener("click", () => {
      if (row.classList.contains("selected")) {
        clearFields();
      } else {
        populateFields(row);
        highlightRow(row);
      }
    });
  }

  function createAndAddRow(data, isNew = true) {
    const newRow = document.createElement("tr");
    newRow.dataset.parentId = data.parent.id;
    newRow.dataset.parentContact = data.parent.contact;
    newRow.dataset.parentPath = data.parent.path || "";  // Add path as data attribute
    newRow.dataset.studentDate = data.student.date;
    newRow.dataset.studentBlastocyste_g1 =
      data.student.blastocyste_g1 || "";
    newRow.dataset.studentBlastocyste_g2 =
      data.student.blastocyste_g2 || "";
    newRow.dataset.studentBlastocyste_g3 =
      data.student.blastocyste_g3 || "";
    newRow.dataset.studentAneuploidy = data.student.aneuploidy || "";
    newRow.dataset.studentLive_birth = data.student.live_birth || "";
    newRow.dataset.annotations = data.annotations
      ? JSON.stringify(data.annotations)
      : "[]";

    const gradingDisplay =
      [
        data.student.blastocyste_g1,
        data.student.blastocyste_g2,
        data.student.blastocyste_g3,
      ]
        .filter((grade) => grade && grade !== "")
        .join("/") || "-";

    newRow.innerHTML = `
              <td>${data.parent.id}</td>
              <td>${data.student.date}</td>
              <td>${data.parent.contact}</td>
              <td>${gradingDisplay}</td>
              <td>${data.student.aneuploidy || "-"}</td>
              <td>${data.student.live_birth || "-"}</td>`;

    addRowClickListener(newRow);
    allDataRows.push(newRow);
    if (isNew) {
      updateNextId();
    }
  }

  function updateRecord() {
    if (!currentRecordId) {
      showModal("Error", "Please select a record from the list to update.");
      return;
    }
    
    if (!verifyInputs()) return;

    const selectedRow = document.querySelector(
      ".data-table tbody .selected"
    );
    if (!selectedRow) {
      return;
    }

    selectedRow.dataset.parentContact = Parent_Contact.value;
    selectedRow.dataset.studentDate = Record_Date.value;
    selectedRow.dataset.studentBlastocyste_g1 = blastocysteGrade1.value;
    selectedRow.dataset.studentBlastocyste_g2 = blastocysteGrade2.value;
    selectedRow.dataset.studentBlastocyste_g3 = blastocysteGrade3.value;
    selectedRow.dataset.studentAneuploidy = pgtaSelect.value;
    selectedRow.dataset.studentLive_birth = liveBirthSelect.value;

    const gradingDisplay =
      [
        blastocysteGrade1.value,
        blastocysteGrade2.value,
        blastocysteGrade3.value,
      ]
        .filter((grade) => grade && grade !== "")
        .join("/") || "-";

    selectedRow.innerHTML = `
              <td>${Parent_ID.value}</td>
              <td>${Record_Date.value}</td>
              <td>${Parent_Contact.value}</td>
              <td>${gradingDisplay}</td>
              <td>${pgtaSelect.value || "-"}</td>
              <td>${liveBirthSelect.value || "-"}</td>`;

    clearFields();
    displayPage();
  }

  async function deleteRecord() {
    if (!currentRecordId) {
      showModal("Error", "Please select a record from the list to delete.");
      return;
    }
    
    const selectedRow = document.querySelector(
      ".data-table tbody .selected"
    );
    if (!selectedRow) {
      showModal("Error", "Please select a record from the list to delete.");
      return;
    }

    const parentIdToDelete = selectedRow.dataset.parentId;
    
    // Show confirmation modal
    showModal(
      "Confirm Delete", 
      `Are you sure you want to delete embryo record ${parentIdToDelete}? This will permanently delete the record and all associated images.`,
      [
        { text: 'Cancel', class: 'btn-secondary', onClick: () => modal.classList.remove('show') },
        { text: 'Delete', class: 'btn-delete', onClick: async () => {
          modal.classList.remove('show');
          await performDelete(parentIdToDelete);
        }}
      ]
    );
  }

  async function performDelete(embryoId) {
    try {
      const response = await fetch('/Doctor/Embryo/DELETE', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          embryo_id: embryoId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      showModal("Success", "Embryo record deleted successfully!");
      
      // Refresh the table
      await fetchAllEmbryo();
    clearFields();
      
    } catch (error) {
      showModal("Error", `Failed to delete embryo record: ${error.message}`);
    }
  }

  async function updateRecord() {
    if (!currentRecordId) {
      showModal("Error", "Please select a record from the list to update.");
      return;
    }
    
    if (!verifyInputs()) {
      return;
    }

    try {
      // Create FormData for update (no images)
      const formData = new FormData();
      
      // Add embryo data
      formData.append('embryo_id', currentRecordId);
      formData.append('date', Record_Date.value);
      formData.append('contact', Parent_Contact.value || '');
      formData.append('blastocyst_grade', `${blastocysteGrade1.value}${blastocysteGrade2.value}${blastocysteGrade3.value}`);
      formData.append('pgt_a_grade', pgtaSelect.value || '');
      formData.append('live_birth', liveBirthSelect.value === 'Yes' ? 'true' : (liveBirthSelect.value === 'No' ? 'false' : ''));
      
      // Add annotations CSV data (only if we have loaded images)
      if (frameImages.length > 0) {
        const csvData = frameImages.map((image, index) => ({
          image_name: image.name,
          phase: frameAnnotations[index] || 'N/A'
        }));
        formData.append('annotations', JSON.stringify(csvData));
      } else {
        // If no images loaded, send empty annotations
        formData.append('annotations', JSON.stringify([]));
      }

      const response = await fetch('/Doctor/Embryo/UPDATE', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      showModal("Success", "Embryo record updated successfully!");
      
      // Refresh the table
      await fetchAllEmbryo();
      clearFields();
      
    } catch (error) {
      showModal("Error", `Failed to update embryo record: ${error.message}`);
    }
  }

  async function loadEmbryoImagesAndAnnotations(embryoId) {
    try {
      // Get image list and annotations from server
      const response = await fetch('/Doctor/Embryo/GET_IMAGES', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          embryo_id: embryoId
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      
      if (result.error) {
        throw new Error(result.error);
      }

      // Clear existing frames
      clearFrames();
      
      // Load images
      if (result.image_files && result.image_files.length > 0) {
        for (let i = 0; i < result.image_files.length; i++) {
          const filename = result.image_files[i];
          const imageUrl = `/Doctor/Embryo/IMAGE/${embryoId}/${filename}`;
          
          // Create a mock file object for compatibility with existing code
          const mockFile = {
            name: filename,
            webkitRelativePath: filename
          };
          
          frameImages.push(mockFile);
          frameUrls.push(imageUrl);
          
          // Create thumbnail
          const thumbItem = document.createElement("div");
          thumbItem.className = "thumbnail-item";
          const currentIndex = frameUrls.length - 1;
          const displayNumber = currentIndex + 1;
          thumbItem.setAttribute("data-index", displayNumber);
          thumbItem.setAttribute("data-internal-index", currentIndex);
          
          const img = document.createElement("img");
          img.src = imageUrl;
          img.alt = `Frame ${displayNumber}`;
          thumbItem.appendChild(img);
          
          const frameNumber = document.createElement("div");
          frameNumber.className = "frame-number";
          frameNumber.textContent = displayNumber;
          thumbItem.appendChild(frameNumber);
          
          thumbnailStrip.appendChild(thumbItem);
        }
        
        // Set up thumbnail click handlers
        const thumbnails = thumbnailStrip.querySelectorAll('.thumbnail-item');
        thumbnails.forEach(thumbItem => {
          thumbItem.addEventListener('click', () => {
            const clickedIndex = parseInt(thumbItem.getAttribute("data-internal-index"), 10);
            showFrame(clickedIndex);
          });
        });
        
        // Show first frame
        if (frameUrls.length > 0) {
          showFrame(0);
          
          // Enable the phase selector
          const phaseSelector = document.getElementById('phase-selector');
          if (phaseSelector) {
            phaseSelector.disabled = false;
            
            // Add event listener for phase changes
            phaseSelector.addEventListener('change', function() {
              const newPhase = this.value;
              
              // Apply phase to ALL frames of the selected embryo
              for (let i = 0; i < frameAnnotations.length; i++) {
                frameAnnotations[i] = newPhase;
              }
              
            });
          }
        }
        
        // Update upload box styling
        uploadBox.classList.add("has-files");
      }
      
      // Load annotations
      if (result.annotations && result.annotations.length > 0) {
        frameAnnotations = new Array(frameImages.length).fill('N/A');
        
        result.annotations.forEach(annotation => {
          const imageName = annotation.image_name;
          const phase = annotation.phase;
          
          // Find the index of this image in our frameImages array
          const imageIndex = frameImages.findIndex(img => img.name === imageName);
          if (imageIndex !== -1) {
            frameAnnotations[imageIndex] = phase;
          }
        });
        
        // Update phase dropdown with the first frame's phase
        const phaseSelector = document.getElementById('phase-selector');
        if (phaseSelector && frameAnnotations.length > 0) {
          phaseSelector.value = frameAnnotations[0] || 'N/A';
        }
      }
      
    } catch (error) {
      showModal("Error", `Failed to load images: ${error.message}`);
    }
  }

  function filterTable() {
    clearFields();
    sortColumn = null;
    sortDirection = null;
    updateSortIcons();
    currentPage = 1;
    displayPage();
  }

  // --- PREDICTION LOGIC ---
  /**
   * AI Prediction Function for Developmental Transitions
   * 
   * This function handles the AI-powered prediction of developmental transitions
   * in embryo image sequences. It processes uploaded images through the backend
   * AI models and displays the results with visual indicators.
   * 
   * Process:
   * 1. Validates that images are uploaded
   * 2. Checks if sufficient frames are available for prediction
   * 3. Converts images to base64 format for API transmission
   * 4. Sends prediction request to backend with frame count
   * 5. Processes prediction results and updates UI
   * 6. Displays visual indicators on thumbnails
   * 
   * Visual Indicators:
   * - Blue circle: No transition predicted
   * - Red circle: Transition predicted
   * - First window: Colors all frames in the window
   * - Subsequent windows: Colors only the newly added frame
   * 
   * Error Handling:
   * - Shows modal for insufficient images
   * - Handles API errors gracefully
   * - Displays user-friendly error messages
   * 
   * @async
   * @function predictLiveBirth
   * @returns {Promise<void>}
   */
  async function predictLiveBirth() {
    // Check if images are uploaded
    if (frameImages.length === 0) {
      showModal(
        "No Images Uploaded",
        "Please upload embryo images before running prediction.",
        [{ text: 'OK', class: 'btn-add' }]
      );
      return;
    }

    const parameter = parseInt(predictParameter.value);
    
    // Check if we have enough frames for the selected parameter
    if (frameImages.length < parameter) {
      showModal(
        "Insufficient Frames",
        `Please upload at least ${parameter} frames for prediction. Currently have ${frameImages.length} frames.`,
        [{ text: 'OK', class: 'btn-add' }]
      );
      return;
    }

    
    try {
      // Convert File objects to Image elements first
      const imagePromises = frameImages.map((file, index) => {
        return new Promise((resolve) => {
          const img = new Image();
          img.onload = () => {
            const canvas = document.createElement('canvas');
            const ctx = canvas.getContext('2d');
            canvas.width = 224;
            canvas.height = 224;
            ctx.drawImage(img, 0, 0, 224, 224);
            const base64 = canvas.toDataURL('image/jpeg').split(',')[1];
            resolve(base64);
          };
          img.src = URL.createObjectURL(file);
        });
      });
      
      const base64Images = await Promise.all(imagePromises);
      
      // Send to server
      const response = await fetch('/Doctor/Embryo/PREDICT', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          frame_count: parameter,
          images: base64Images
        }),
        signal: AbortSignal.timeout(30000) // 30 second timeout
      });
      
      const result = await response.json();
      
      if (result.success) {
        displayTransitionPredictions(result.predictions, parameter);
        
        // Check if predictions are random (model not found)
        if (result.is_random) {
          showModal(
            "Random Predictions Used",
            `Model not found! Using random predictions for demonstration. Results are not accurate.`,
            [{ text: 'OK', class: 'btn-add' }]
          );
        } else {
          showModal(
            "Prediction Complete",
            `Transition analysis completed for ${parameter} frames using AI model. Check frame indicators for results.`,
            [{ text: 'OK', class: 'btn-add' }]
          );
        }
      } else {
        showModal(
          "Prediction Failed",
          result.message || "Failed to predict transitions",
          [{ text: 'OK', class: 'btn-add' }]
        );
      }
      
    } catch (error) {
      showModal(
        "Prediction Error",
        "Failed to predict transitions. Please try again.",
        [{ text: 'OK', class: 'btn-add' }]
      );
    }
  }

  function displayTransitionPredictions(predictions, frameCount) {
    // Clear previous predictions
    clearTransitionIndicators();
    
    // Apply predictions with sliding window approach
    for (let i = 0; i < predictions.length; i++) {
      const prediction = predictions[i];
      
      if (i === 0) {
        // First window: color ALL frames in the window
        for (let frameIdx = 0; frameIdx < frameCount; frameIdx++) {
          if (frameIdx < frameImages.length) {
            addTransitionIndicator(frameIdx, prediction);
          }
        }
      } else {
        // Subsequent windows: color only the NEW frame (last frame in window)
        const newFrameIndex = i + frameCount - 1;
        if (newFrameIndex < frameImages.length) {
          addTransitionIndicator(newFrameIndex, prediction);
        }
      }
    }
  }

  function addTransitionIndicator(frameIndex, prediction) {
    // Find the thumbnail for this frame
    const thumbnail = thumbnailStrip.querySelector(`[data-internal-index="${frameIndex}"]`);
    if (!thumbnail) return;
    
    // Create or update transition indicator
    let indicator = thumbnail.querySelector('.transition-indicator');
    if (!indicator) {
      indicator = document.createElement('div');
      indicator.className = 'transition-indicator';
      thumbnail.appendChild(indicator);
    }
    
    // Set color based on prediction (0 = blue/no transition, 1 = red/transition)
    if (prediction === 0) {
      indicator.style.backgroundColor = '#007bff'; // Blue
      indicator.title = 'No transition detected';
    } else {
      indicator.style.backgroundColor = '#dc3545'; // Red
      indicator.title = 'Transition detected';
    }
  }

  function clearTransitionIndicators() {
    const indicators = thumbnailStrip.querySelectorAll('.transition-indicator');
    indicators.forEach(indicator => indicator.remove());
  }

  // --- FRAME PREVIEW SYSTEM ---
  function clearFrames() {
    frameUrls.forEach((url) => URL.revokeObjectURL(url));
    frameImages = [];
    frameUrls = [];
    frameAnnotations = [];
    currentFrameIndex = 0;
    
    thumbnailStrip.innerHTML = "";
    currentImage.src = "https://placehold.co/400x300/f0f0f0/cccccc?text=Preview";
    frameText.textContent = "Frame 0 of 0";
    phaseSelector.value = "N/A";
    phaseSelector.disabled = true;
    uploadBox.classList.remove("has-files");
    
    // Re-enable phase selector
    phaseSelector.disabled = false;
    
    if (playbackInterval) {
      clearInterval(playbackInterval);
      isPlaying = false;
    }
    playBtn.innerHTML = '<i class="fas fa-play"></i>';
    playBtn.classList.remove("playing");
    updateButtonStates();
  }

  function handleFrameUpload(event) {
    const files = Array.from(event.target.files).sort(naturalSort);
    if (files.length === 0) return;
    
    clearFrames();
    
    files.forEach((file) => {
      if (file.type.startsWith("image/")) {
        frameImages.push(file);
        const url = URL.createObjectURL(file);
        frameUrls.push(url);
        
         const thumbItem = document.createElement("div");
         thumbItem.className = "thumbnail-item";
         const currentIndex = frameUrls.length - 1;
         const displayNumber = currentIndex + 1; // 1-based display number
         thumbItem.setAttribute("data-index", displayNumber); // Set 1-based for display
         thumbItem.setAttribute("data-internal-index", currentIndex); // Set 0-based for internal logic
         
         const thumbImg = document.createElement("img");
         thumbImg.src = url;
         thumbItem.appendChild(thumbImg);
         
         thumbItem.addEventListener("click", () => {
           const clickedIndex = parseInt(thumbItem.getAttribute("data-internal-index"), 10);
           const displayNumber = thumbItem.getAttribute("data-index");
           showFrame(clickedIndex);
         });
         
         thumbnailStrip.appendChild(thumbItem);
      }
    });

    if (frameUrls.length > 0) {
      frameAnnotations = new Array(frameUrls.length).fill("N/A");
      phaseSelector.disabled = false;
      updateButtonStates();
      uploadBox.classList.add("has-files");
      
      // Small delay to ensure thumbnails are rendered before showing first frame
      setTimeout(() => {
        showFrame(0);
        // Ensure the first thumbnail is marked as active
        const firstThumb = thumbnailStrip.querySelector(`[data-index="1"]`);
        if (firstThumb) {
          firstThumb.classList.add("active");
        }
      }, 100);
    }
  }

  function showFrame(index) {
    // Ensure index is a number
    index = parseInt(index, 10);
    if (index >= 0 && index < frameUrls.length) {
      currentFrameIndex = index;
      currentImage.src = frameUrls[index];
      const frameNumber = index + 1;
      const totalFrames = frameUrls.length;
      frameText.textContent = `Frame ${frameNumber} of ${totalFrames}`;
      phaseSelector.value = frameAnnotations[index];

      thumbnailStrip.querySelectorAll(".thumbnail-item").forEach((item) => {
        item.classList.remove("active");
      });
      
      const activeThumb = thumbnailStrip.querySelector(`[data-index="${index + 1}"]`);
      if (activeThumb) {
        activeThumb.classList.add("active");
        // Scroll the active thumbnail into view
        activeThumb.scrollIntoView({
          behavior: "smooth",
          block: "nearest",
          inline: "nearest"
        });
      }
      
      updateButtonStates();
    }
  }

  function updateButtonStates() {
    prevBtn.disabled = currentFrameIndex === 0;
    nextBtn.disabled = currentFrameIndex === frameUrls.length - 1;
    playBtn.disabled = frameUrls.length <= 1;
  }

  function goToPreviousFrame() {
    if (currentFrameIndex > 0) {
      showFrame(currentFrameIndex - 1);
    }
  }

  function goToNextFrame() {
    if (currentFrameIndex < frameUrls.length - 1) {
      showFrame(currentFrameIndex + 1);
    }
  }

  function togglePlayback() {
    if (isPlaying) {
      clearInterval(playbackInterval);
      isPlaying = false;
      playBtn.innerHTML = '<i class="fas fa-play"></i>';
      playBtn.classList.remove("playing");
    } else {
      isPlaying = true;
      playBtn.innerHTML = '<i class="fas fa-pause"></i>';
      playBtn.classList.add("playing");
      playbackInterval = setInterval(() => {
        if (currentFrameIndex < frameUrls.length - 1) {
          showFrame(currentFrameIndex + 1);
        } else {
          togglePlayback();
        }
      }, 800);
    }
  }

  function handlePhaseChange(event) {
    const newPhase = event.target.value;
    for (let i = currentFrameIndex; i < frameAnnotations.length; i++) {
      frameAnnotations[i] = newPhase;
    }
  }

  function handleArrowKeys(event) {
    // Don't handle arrow keys if a dropdown/select is focused
    if (event.target.tagName === 'SELECT' || event.target.classList.contains('phase-dropdown')) {
      return;
    }
    
    if (frameUrls.length === 0) return;
    if (event.key === "ArrowLeft" && currentFrameIndex > 0) {
      showFrame(currentFrameIndex - 1);
    } else if (event.key === "ArrowRight" && currentFrameIndex < frameUrls.length - 1) {
      showFrame(currentFrameIndex + 1);
    }
  }

  async function addRecord() {
    if (!verifyInputs()) {
      return;
    }

    // Always generate new ID for new records (ID field is always empty and read-only)
    const recordId = nextParentId;
    const recordDate = Record_Date.value;

    try {
      // Create FormData for file upload
      const formData = new FormData();
      
      // Add embryo data
      formData.append('embryo_id', recordId);
      formData.append('date', recordDate);
      formData.append('contact', Parent_Contact.value || '');
      formData.append('blastocyst_grade', `${blastocysteGrade1.value}${blastocysteGrade2.value}${blastocysteGrade3.value}`);
      formData.append('pgt_a_grade', pgtaSelect.value || '');
      formData.append('live_birth', liveBirthSelect.value === 'Yes' ? 'true' : (liveBirthSelect.value === 'No' ? 'false' : ''));
      
      // Add images
      frameImages.forEach((image, index) => {
        formData.append('images', image);
      });
      
      // Add annotations CSV data
      const csvData = frameImages.map((image, index) => {
        return {
          image_name: image.name, // This should already be just the filename
          phase: frameAnnotations[index] || 'N/A'
        };
      });
      formData.append('annotations', JSON.stringify(csvData));

      const response = await fetch('/Doctor/Embryo/ADD', {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Server error');
      }

      const result = await response.json();
      
      showModal("Success", "Embryo record added successfully!");
      
      // Refresh the table
      await fetchAllEmbryo();
      clearFields();
      
    } catch (error) {
      showModal("Error", `Failed to add embryo record: ${error.message}`);
    }
  }

  // --- DATABASE FUNCTIONS ---
  async function fetchAllEmbryo() {
    try {
      const response = await fetch('/Doctor/Embryo/LIST');
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const embryoData = await response.json();
      
      // Clear existing data
      allDataRows = [];
      
      // Convert embryo data to table format and add rows
      embryoData.forEach(embryo => {
        const tableData = {
      parent: {
            id: embryo.embryo_id,
            contact: embryo.contact || "",
            path: embryo.path || ""  // Add path field
      },
      student: {
            id: embryo.embryo_id,
            date: embryo.date || "",
            blastocyste_g1: embryo.blastocyst_grade ? embryo.blastocyst_grade.charAt(0) : "",
            blastocyste_g2: embryo.blastocyst_grade ? embryo.blastocyst_grade.charAt(1) : "",
            blastocyste_g3: embryo.blastocyst_grade ? embryo.blastocyst_grade.charAt(2) : "",
            aneuploidy: embryo.pgt_a_grade || "",
            live_birth: embryo.live_birth === true ? "Yes" : (embryo.live_birth === false ? "No" : "")
      },
      annotations: [] // Will be populated later if needed
        };
        
        createAndAddRow(tableData, false);
      });
      
      // Update display
      updateNextId();
    displayPage();
      
      return true;
    } catch (error) {
      return false;
    }
  }

  // --- MODAL FUNCTIONALITY ---
  function showModal(title, message, buttons = [{text: 'OK', class: 'btn-add'}]) {
    modalTitle.textContent = title;
    modalMessage.textContent = message;
    modalButtons.innerHTML = '';

    buttons.forEach(btnInfo => {
      const button = document.createElement('button');
      button.textContent = btnInfo.text;
      button.className = `btn ${btnInfo.class}`;
      button.onclick = () => {
        modal.classList.remove('show');
        if (btnInfo.onClick) {
          btnInfo.onClick();
        }
      };
      modalButtons.appendChild(button);
    });
    
    modal.classList.add('show');
    
    // Add default OK button behavior if no custom buttons
    if (buttons.length === 1 && buttons[0].text === 'OK') {
      const okButton = modalButtons.querySelector('button');
      okButton.onclick = () => modal.classList.remove('show');
    }
  }

  // --- LOGOUT FUNCTIONALITY ---
  function handleLogout() {
    showModal(
      "Confirm Logout", 
      "Are you sure you want to logout?", 
      [
        { text: 'Cancel', class: 'btn-clear' },
        { text: 'Logout', class: 'btn-delete', onClick: performLogout }
      ]
    );
  }

  function performLogout() {
    // Clear any local data
    clearFields();
    allDataRows = [];
    
    // Redirect to logout endpoint
    window.location.href = '/Log_Out';
  }

  // --- EVENT LISTENERS ---
  clearButton.addEventListener("click", clearFields);
  addButton.addEventListener("click", addRecord);
  updateButton.addEventListener("click", updateRecord);
  deleteButton.addEventListener("click", deleteRecord);
  searchInput.addEventListener("input", filterTable);
  tableHeader.addEventListener("click", (e) => {
    const th = e.target.closest("th");
    if (th && th.dataset.column) sortTable(th.dataset.column);
  });
  entriesPerPageSelect.addEventListener("change", () => {
    rowsPerPage = parseInt(entriesPerPageSelect.value, 10);
    currentPage = 1;
    displayPage();
  });
  prevButton.addEventListener("click", () => {
    if (currentPage > 1) {
      currentPage--;
      displayPage();
    }
  });
  nextButton.addEventListener("click", () => {
    if (currentPage < Math.ceil(getFilteredRows().length / rowsPerPage)) {
      currentPage++;
      displayPage();
    }
  });

  predictButton.addEventListener("click", predictLiveBirth);
  phaseSelector.addEventListener("change", handlePhaseChange);

  prevBtn.addEventListener("click", goToPreviousFrame);
  playBtn.addEventListener("click", togglePlayback);
  nextBtn.addEventListener("click", goToNextFrame);

  uploadBox.addEventListener("click", (event) => {
    // Don't trigger upload if clicking on thumbnails
    if (event.target.closest('.thumbnail-strip')) {
      return;
    }
    
    // Don't trigger upload if files are already uploaded
    if (uploadBox.classList.contains('has-files')) {
      return;
    }
    
    const input = document.createElement("input");
    input.type = "file";
    input.webkitdirectory = true;
    input.directory = true;
    input.multiple = true;
    input.style.display = "none";
    input.addEventListener("change", handleFrameUpload);
    document.body.appendChild(input);
    input.click();
    input.remove();
  });

  document.addEventListener("keydown", handleArrowKeys);

  // Logout button event listener
  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) {
    logoutBtn.addEventListener("click", (e) => {
      e.preventDefault(); // Prevent default link behavior
      handleLogout();
    });
  }

  // --- INITIAL SETUP ---
  function initializeTableSorting() {
    sortColumn = null;
    sortDirection = null;
    updateSortIcons();
  }

  initializeTableSorting();
  clearFields();
  displayPage();
  
  // Load embryo data from database on page load
  fetchAllEmbryo();
});