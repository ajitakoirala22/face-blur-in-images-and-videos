const form = document.getElementById("uploadForm");
const statusEl = document.getElementById("status");
const submitBtn = document.getElementById("submitBtn");
const resultSection = document.getElementById("result");
const preview = document.getElementById("preview");
const downloadLink = document.getElementById("downloadLink");
const progressWrap = document.getElementById("progressWrap");
const progressFill = document.getElementById("progressFill");
const progressText = document.getElementById("progressText");

function setProgress(value, message = "") {
  const pct = Math.max(0, Math.min(100, Number(value) || 0));
  progressFill.style.width = `${pct}%`;
  progressText.textContent = message ? `${pct}% - ${message}` : `${pct}%`;
}

async function pollStatus(jobId) {
  while (true) {
    const res = await fetch(`/status/${jobId}`);
    const data = await res.json();

    if (!res.ok) {
      throw new Error(data.error || "Failed to fetch job status");
    }

    setProgress(data.progress, data.message || "Processing");

    if (data.status === "completed") {
      return data;
    }
    if (data.status === "failed") {
      throw new Error(data.message || "Processing failed");
    }

    await new Promise((resolve) => setTimeout(resolve, 500));
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  statusEl.textContent = "Processing...";
  submitBtn.disabled = true;
  progressWrap.classList.remove("hidden");
  setProgress(0, "Queued");
  resultSection.classList.add("hidden");
  preview.innerHTML = "";

  const formData = new FormData();
  const fileInput = document.getElementById("file");
  const confInput = document.getElementById("confidence");
  const enhanceInput = document.getElementById("enhance");
  const deviceInput = document.getElementById("device");

  if (!fileInput.files.length) {
    statusEl.textContent = "Please select a file.";
    submitBtn.disabled = false;
    return;
  }

  formData.append("file", fileInput.files[0]);
  formData.append("confidence", confInput.value);
  formData.append("enhance", enhanceInput.checked ? "true" : "false");
  formData.append("device", deviceInput.value);

  try {
    const res = await fetch("/process", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();

    if (!res.ok || !data.ok || !data.job_id) {
      throw new Error(data.error || "Could not start processing job");
    }

    const finalData = await pollStatus(data.job_id);

    statusEl.textContent = `Done. Device used: ${finalData.device_used || "cpu"}`;
    downloadLink.href = finalData.output_url;
    downloadLink.setAttribute("download", finalData.download_name);

    if (finalData.type === "image") {
      const img = document.createElement("img");
      img.src = finalData.output_url;
      img.alt = "Blurred output";
      preview.appendChild(img);
    } else {
      const video = document.createElement("video");
      video.src = finalData.output_url;
      video.controls = true;
      video.playsInline = true;
      preview.appendChild(video);
    }

    resultSection.classList.remove("hidden");
  } catch (err) {
    statusEl.textContent = `Failed: ${err.message}`;
    progressWrap.classList.add("hidden");
  } finally {
    submitBtn.disabled = false;
  }
});
