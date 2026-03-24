/* ============================================================
   Two-Phase Pending Modals
   Phase 1: Uploading (DO NOT CLOSE)
   Phase 2: Running / Queue (May Leave Page)
============================================================ */

/* ---------- Phase 1: Uploading ---------- */
function showUploadingModal() {
    const modal = document.getElementById("uploadingModal");
    modal.style.display = "flex";

    // 禁止關閉頁面（避免上傳被中斷）
    window.onbeforeunload = () => "Your file is still uploading. Do NOT close this page.";
}

function hideUploadingModal() {
    const modal = document.getElementById("uploadingModal");
    modal.style.display = "none";
    window.onbeforeunload = null;  // 上傳完成 → 開放關閉頁面
}

/* ---------- Phase 2: Running / Queue ---------- */
function showPendingModal(jobId) {
    const modal = document.getElementById("pendingModal");
    const spanId = document.getElementById("pendingJobId");
    spanId.textContent = jobId;
    modal.style.display = "flex";
}

function hidePendingModal() {
    const modal = document.getElementById("pendingModal");
    modal.style.display = "none";
}


/* ============================================================
   Restore pending on reload (Phase 2 only)
============================================================ */
async function restorePendingIfNeeded() {
    const params = new URLSearchParams(window.location.search);
    const jobId = params.get("job_id");

    if (!jobId) return;

    const resp = await fetch(`/api/job/${jobId}`);
    const data = await resp.json();

    if (data.found) {
        // 已完成 → 直接跳結果
        window.location.href = `/results?job_id=${jobId}`;
        return;
    }

    // 尚未完成 → 顯示 Phase 2 pending
    showPendingModal(jobId);

    const timer = setInterval(async () => {
        const r = await fetch(`/api/job/${jobId}`);
        const d = await r.json();

        if (d.found) {
            clearInterval(timer);
            hidePendingModal();
            window.location.href = `/results?job_id=${jobId}`;
        }
    }, 2000);
}


/* ============================================================
   Submit: register → show uploading → await upload → pending
============================================================ */
async function handleSubmit() {
    const fileInput = document.getElementById("fileInput");
    const file = fileInput.files[0];
    const exampleFlag = document.getElementById("example_flag").value;

    /* ---------------- Step 1. Register job (instant) ---------------- */
    const formData = new FormData();
    formData.append("tau1", document.getElementById("tau1").value);
    formData.append("tau2", document.getElementById("tau2").value);
    formData.append("index", document.getElementById("index").value);
    formData.append("label", document.getElementById("label").value);
    formData.append("gmail", document.getElementById("gmail").value);
    formData.append("example_flag", exampleFlag);

    const reg = await fetch("/api/register-job", {
        method: "POST",
        body: formData
    });

    const regJson = await reg.json();
    if (!regJson.success) {
        alert("Register failed: " + (regJson.error || "Unknown error"));
        return;
    }

    const jobId = regJson.job_id;

    /* ---------------- Step 2. Show Uploading Modal ---------------- */
    showUploadingModal();

    /* ---------------- Step 3. Upload the file (MUST await!) ---------------- */
    const uploadData = new FormData();
    if (exampleFlag !== "true") {
        uploadData.append("file", file);
    }

    const uploadResp = await fetch(`/api/upload/${jobId}`, {
        method: "POST",
        body: uploadData
    });

    if (!uploadResp.ok) {
        hideUploadingModal();
        alert("File upload failed!");
        return;
    }

    /* ---------------- Step 4. Upload complete → switch to Phase 2 ---------------- */
    hideUploadingModal();

    showPendingModal(jobId);

    // 可 bookmark，此時允許離開頁面
    history.pushState({}, "", `/?job_id=${jobId}`);

    /* ---------------- Step 5. Start polling ---------------- */
    const timer = setInterval(async () => {
        const resp = await fetch(`/api/job/${jobId}`);
        const data = await resp.json();

        if (data.found) {
            clearInterval(timer);
            hidePendingModal();
            window.location.href = `/results?job_id=${jobId}`;
        }
    }, 2000);
}


/* ============================================================
   DOM Ready
============================================================ */
document.addEventListener("DOMContentLoaded", function () {

    const submitBtn = document.getElementById("submitBtn");
    submitBtn.addEventListener("click", handleSubmit);

    // 若 URL 帶 job_id → Phase 2 還原
    restorePendingIfNeeded();
});




