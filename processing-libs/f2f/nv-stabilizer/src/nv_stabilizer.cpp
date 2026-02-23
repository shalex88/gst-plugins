#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>

#include <vpi/VPI.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/PerspectiveWarp.h>
#include <vpi/algo/HarrisCorners.h>
#include <vpi/algo/KLTFeatureTracker.h>
#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/ImageFormat.h>
#include <vpi/Status.h>
#include <vpi/Stream.h>

#include "processor_api.h"

#define CHECK_STATUS(STMT)                                    \
    do                                                        \
    {                                                         \
        VPIStatus status = (STMT);                            \
        if (status != VPI_SUCCESS)                            \
        {                                                     \
            char buffer[VPI_MAX_STATUS_MESSAGE_LENGTH];       \
            vpiGetLastStatusMessage(buffer, sizeof(buffer));  \
            std::ostringstream ss;                            \
            ss << "line " << __LINE__ << ": ";                \
            ss << vpiStatusGetName(status) << ": " << buffer; \
            throw std::runtime_error(ss.str());               \
        }                                                     \
    } while (0);

#define MOTION_HISTORY_SIZE 15

typedef struct {
    uint64_t frame_count;
    const char* cfg_path_seen;

    int width;
    int height;
    int stride;
    bool dims_valid;

    VPIStream vpi_stream;

    // VPI-owned images (Y and UV planes)
    VPIImage cur_img_y;
    VPIImage prev_img_y;
    VPIImage out_img_y;

    VPIImage cur_img_uv;
    VPIImage prev_img_uv;
    VPIImage out_img_uv;

    // Harris corner detection
    VPIPayload harris_payload;
    VPIArray keypoints_cur;        // Harris output keypoints (VPI_ARRAY_TYPE_KEYPOINT_F32)
    VPIArray harris_scores;        // Harris output scores    (VPI_ARRAY_TYPE_F32)
    VPIHarrisCornerDetectorParams harris_params;

    // KLT feature tracking
    // keypoints_prev    — reference boxes in the template frame  (KLT_TRACKED_BOUNDING_BOX)
    // tracked_features  — initial position predictions            (KLT_TRACKED_BOUNDING_BOX)
    // tracking_estimates— KLT output: tracked boxes in cur frame  (KLT_TRACKED_BOUNDING_BOX)
    VPIPayload klt_payload;
    VPIArray keypoints_prev;
    VPIArray tracked_features;
    VPIArray tracking_estimates;
    VPIKLTFeatureTrackerParams klt_params;

    // Motion estimation
    float affine_matrix[6];        // [a b tx c d ty] — raw per-frame camera delta
    float smoothed_affine[6];      // Warp to apply this frame (correction)

    // Trajectory-based smoothing
    // traj = accumulated raw camera path; smooth_traj = low-pass of traj
    // correction = smooth_traj - traj  (how far to shift to follow smooth path)
    float trajectory[2];           // accumulated (tx, ty)
    float smoothed_trajectory[2];  // EMA of trajectory

    // Kept for potential future window-based filter
    float motion_history[MOTION_HISTORY_SIZE][6];
    int history_index;
    bool history_full;

    // State
    bool has_prev_features;
    int num_tracked_points;
    int redetect_counter;

} NvStabCtx;

static ProcStatus nv_stab_init(const char* config_path, void** ctx)
{
    NvStabCtx* c = (NvStabCtx*)malloc(sizeof(NvStabCtx));
    if (!c)
        return PROC_STATUS_ERR_ALLOC;

    memset(c, 0, sizeof(NvStabCtx));

    c->frame_count = 0;
    c->cfg_path_seen = config_path ? strdup(config_path) : nullptr;
    c->dims_valid = false;
    c->has_prev_features = false;
    c->num_tracked_points = 0;
    c->history_index = 0;
    c->history_full = false;
    c->redetect_counter = 0;

    // VPI stream will be created lazily on first frame (thread safety)
    c->vpi_stream = NULL;

    // Harris corner detection parameters
    memset(&c->harris_params, 0, sizeof(c->harris_params));
    c->harris_params.strengthThresh = 0.01f;
    c->harris_params.sensitivity = 0.01f;
    c->harris_params.minNMSDistance = 5;

    // KLT tracking parameters
    c->klt_params.numberOfIterationsScaling = 2;
    c->klt_params.nccThresholdUpdate = 0.7f;
    c->klt_params.trackingType = VPI_KLT_INVERSE_COMPOSITIONAL;

    // Initialize identity affine transform
    c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
    c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;

    memcpy(c->smoothed_affine, c->affine_matrix, sizeof(c->affine_matrix));

    c->trajectory[0]          = 0.0f; c->trajectory[1]          = 0.0f;
    c->smoothed_trajectory[0] = 0.0f; c->smoothed_trajectory[1] = 0.0f;

    *ctx = c;
    //printf("[nv-stabilizer] Initialized (VPI resources will be created lazily on first frame)\n");
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_reset_context(NvStabCtx *c, int w, int h, int stride) {
    //printf("[nv-stabilizer] reset_context check: dims_valid=%d, width=%d/%d, height=%d/%d, stride=%d/%d\n", c->dims_valid, c->width, w, c->height, h, c->stride, stride);

    if (!c->dims_valid || c->width != w || c->height != h || c->stride != stride) {
        //printf("[nv-stabilizer] Resetting context for new dimensions\n");

        c->width = w;
        c->height = h;
        c->stride = stride;
        c->dims_valid = true;

        // Create VPI stream if not already created (lazy init for thread safety)
        if (!c->vpi_stream) {
            //printf("[nv-stabilizer] Creating VPI stream (lazy init)\n");
            VPIStatus st = vpiStreamCreate(0, &c->vpi_stream);
            if (st != VPI_SUCCESS) {
                //printf("[nv-stabilizer] vpiStreamCreate failed: %d\n", (int)st);
                return PROC_STATUS_ERR_GENERAL;
            }
        }

        // Destroy old Y plane images
        if (c->cur_img_y) vpiImageDestroy(c->cur_img_y);
        if (c->prev_img_y) vpiImageDestroy(c->prev_img_y);
        if (c->out_img_y) vpiImageDestroy(c->out_img_y);

        // Destroy old UV plane images
        if (c->cur_img_uv) vpiImageDestroy(c->cur_img_uv);
        if (c->prev_img_uv) vpiImageDestroy(c->prev_img_uv);
        if (c->out_img_uv) vpiImageDestroy(c->out_img_uv);

        // Create Y plane images
        //printf("[nv-stabilizer] Creating Y images %dx%d\n", w, h);
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8, 0, &c->cur_img_y));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8, 0, &c->prev_img_y));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8, 0, &c->out_img_y));
        //printf("[nv-stabilizer] Y images created: cur=%p prev=%p out=%p\n", c->cur_img_y, c->prev_img_y, c->out_img_y);

        // UV plane images disabled - not used in current stabilization implementation
        //printf("[nv-stabilizer] Skipping UV image creation (Y-plane only stabilization)\n");
        c->cur_img_uv = NULL;
        c->prev_img_uv = NULL;
        c->out_img_uv = NULL;

        // Destroy old Harris/KLT resources
        if (c->harris_payload) vpiPayloadDestroy(c->harris_payload);
        if (c->klt_payload) vpiPayloadDestroy(c->klt_payload);
        if (c->keypoints_cur) vpiArrayDestroy(c->keypoints_cur);
        if (c->harris_scores) vpiArrayDestroy(c->harris_scores);
        if (c->keypoints_prev) vpiArrayDestroy(c->keypoints_prev);
        if (c->tracked_features) vpiArrayDestroy(c->tracked_features);
        if (c->tracking_estimates) vpiArrayDestroy(c->tracking_estimates);

        // Create arrays
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KEYPOINT_F32,             0, &c->keypoints_cur));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_F32,                      0, &c->harris_scores));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &c->keypoints_prev));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &c->tracked_features));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &c->tracking_estimates));

        // Create Harris corner detector
        VPIStatus harris_st = vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, w, h, &c->harris_payload);
        if (harris_st != VPI_SUCCESS) {
            c->harris_payload = NULL;
        }

        // Create KLT feature tracker
        CHECK_STATUS(vpiCreateKLTFeatureTracker(VPI_BACKEND_CUDA, w, h,
                                                 VPI_IMAGE_FORMAT_Y8, 0, &c->klt_payload));

        c->has_prev_features = false;

        // Reset trajectory on dimension change so the new session starts fresh
        c->trajectory[0]          = 0.0f; c->trajectory[1]          = 0.0f;
        c->smoothed_trajectory[0] = 0.0f; c->smoothed_trajectory[1] = 0.0f;
        //printf("[nv-stabilizer] Context reset: %dx%d, VPI resources initialized on processing thread\n", w, h);
    }
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_prepare_frame(NvStabCtx* c, VP_Frame* frame)
{
    if (!c || !frame) return PROC_STATUS_ERR_GENERAL;
    if (frame->pixfmt != PROC_PIXFMT_NV12) return PROC_STATUS_ERR_UNSUPPORTED;
    if (!frame->data || frame->width <= 0 || frame->height <= 0 || frame->stride <= 0)
        return PROC_STATUS_ERR_GENERAL;

    const int w = frame->width;
    const int h = frame->height;
    const int stride = frame->stride;

    if(nv_stab_reset_context(c, w, h, stride) != PROC_STATUS_OK)
        return PROC_STATUS_ERR_GENERAL;

    //printf("[nv-stabilizer] prepare_frame: w=%d h=%d stride=%d, cur_img_y=%p\n", w, h, stride, c->cur_img_y);

    // Copy Y plane data
    //printf("[nv-stabilizer] Copying Y plane\n");
    {
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(c->cur_img_y, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));

        uint8_t* src = (uint8_t*)frame->data;
        uint8_t* dst = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        int y_size = frame->width * frame->height;
        memcpy(dst, src, y_size);

        vpiImageUnlock(c->cur_img_y);
    }
    //printf("[nv-stabilizer] Y plane copied\n");

    // Copy UV plane data - DISABLED for now due to buffer access issues
    // TODO: Investigate proper VPI UV plane buffer access
    /*
    {
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(c->cur_img_uv, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));

        uint8_t* src = (uint8_t*)frame->data + (frame->width * frame->height);
        uint8_t* dst = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        int uv_size = (frame->width * frame->height) / 2;  // UV plane is half the Y plane size
        memcpy(dst, src, uv_size);

        vpiImageUnlock(c->cur_img_uv);
    }
    */

    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_detect_features(NvStabCtx *c)
{
    int num_kpts = 0;

    // Use Harris corner detector if available; fall back to uniform grid
    if (c->harris_payload) {
        CHECK_STATUS(vpiSubmitHarrisCornerDetector(
            c->vpi_stream, VPI_BACKEND_CUDA, c->harris_payload,
            c->cur_img_y, c->keypoints_cur, c->harris_scores, &c->harris_params));
        CHECK_STATUS(vpiStreamSync(c->vpi_stream));

        VPIArrayData kpData;
        CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &kpData));
        num_kpts = *kpData.buffer.aos.sizePointer;
        vpiArrayUnlock(c->keypoints_cur);
    }

    // Fall back to evenly-spaced grid when Harris finds too few points
    if (!c->harris_payload || num_kpts < 20) {
        VPIArrayData arrInit;
        CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &arrInit));
        VPIKeypointF32* kpts = (VPIKeypointF32*)arrInit.buffer.aos.data;
        num_kpts = 0;
        const int grid_spacing = 80;
        for (int y = 50; y < c->height && num_kpts < 500; y += grid_spacing)
            for (int x = 50; x < c->width && num_kpts < 500; x += grid_spacing) {
                kpts[num_kpts].x = (float)x;
                kpts[num_kpts].y = (float)y;
                num_kpts++;
            }
        *arrInit.buffer.aos.sizePointer = num_kpts;
        vpiArrayUnlock(c->keypoints_cur);
    }

    // Convert keypoints → KLT bounding boxes.
    // keypoints_prev  = reference boxes for the KLT template (cur_img_y; after the
    //                   per-frame swap this will be prev_img_y which the tracker reads).
    // tracked_features = identity predictions (same position — no prior motion known).
    const float BOX = 21.0f;  // 21×21 pixel patch
    const float HALF = BOX / 2.0f;

    VPIArrayData kpData, refData, predData;
    CHECK_STATUS(vpiArrayLockData(c->keypoints_cur,    VPI_LOCK_READ,  VPI_ARRAY_BUFFER_HOST_AOS, &kpData));
    CHECK_STATUS(vpiArrayLockData(c->keypoints_prev,   VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &refData));
    CHECK_STATUS(vpiArrayLockData(c->tracked_features, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &predData));

    VPIKeypointF32*           kpts    = (VPIKeypointF32*)kpData.buffer.aos.data;
    VPIKLTTrackedBoundingBox* ref_bb  = (VPIKLTTrackedBoundingBox*)refData.buffer.aos.data;
    VPIKLTTrackedBoundingBox* pred_bb = (VPIKLTTrackedBoundingBox*)predData.buffer.aos.data;
    int n = *kpData.buffer.aos.sizePointer;

    // VPIBoundingBox layout: top-left position in xform.mat3[r][2] (translation),
    // scale in xform.mat3[0][0] / [1][1], homogeneous in mat3[2][2].
    // Axis-aligned accessors: x = mat3[0][2], y = mat3[1][2],
    //                         w = width*mat3[0][0], h = height*mat3[1][1]
    for (int i = 0; i < n; i++) {
        float bx = kpts[i].x - HALF;
        float by = kpts[i].y - HALF;
        bx = std::max(0.0f, std::min((float)(c->width  - (int)BOX - 1), bx));
        by = std::max(0.0f, std::min((float)(c->height - (int)BOX - 1), by));

        // Initialise both reference and prediction boxes identically
        for (auto* bb : {&ref_bb[i], &pred_bb[i]}) {
            memset(&bb->bbox.xform, 0, sizeof(bb->bbox.xform));
            bb->bbox.xform.mat3[0][0] = 1.0f;  // x scale
            bb->bbox.xform.mat3[1][1] = 1.0f;  // y scale
            bb->bbox.xform.mat3[2][2] = 1.0f;  // homogeneous
            bb->bbox.xform.mat3[0][2] = bx;    // left
            bb->bbox.xform.mat3[1][2] = by;    // top
            bb->bbox.width  = BOX;
            bb->bbox.height = BOX;
            bb->trackingStatus = 0;  // valid
        }
        ref_bb[i].templateStatus  = 1;  // force template extraction on first track
        pred_bb[i].templateStatus = 0;  // prediction: no re-extraction needed
    }
    *refData.buffer.aos.sizePointer  = n;
    *predData.buffer.aos.sizePointer = n;

    vpiArrayUnlock(c->tracked_features);
    vpiArrayUnlock(c->keypoints_prev);
    vpiArrayUnlock(c->keypoints_cur);

    c->num_tracked_points = n;
    printf("[nv-stabilizer] Detected %d features\n", n);
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_track_features(NvStabCtx *c)
{
    if (!c->has_prev_features) {
        // No valid previous frame yet (just after detection or first-ever frame).
        // Reference boxes are already set up in keypoints_prev from detect_features.
        // Mark as ready and wait for the next frame to actually run KLT.
        c->has_prev_features = true;
        return PROC_STATUS_OK;
    }

    // Run KLT on GPU: track reference boxes from prev_img_y into cur_img_y.
    //   templateImage  = prev_img_y   (the frame where reference patches live)
    //   inputBoxList   = keypoints_prev   (reference boxes in template)
    //   inputPredList  = tracked_features (predicted positions in cur frame)
    //   outputBoxList  = tracking_estimates (tracked positions in cur frame)
    CHECK_STATUS(vpiSubmitKLTFeatureTracker(
        c->vpi_stream,
        VPI_BACKEND_CUDA,
        c->klt_payload,
        c->prev_img_y,
        c->keypoints_prev,
        c->tracked_features,
        c->cur_img_y,
        c->tracking_estimates,
        NULL,
        &c->klt_params));
    CHECK_STATUS(vpiStreamSync(c->vpi_stream));

    // Count valid (not lost) features
    VPIArrayData outData;
    CHECK_STATUS(vpiArrayLockData(c->tracking_estimates, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outData));
    VPIKLTTrackedBoundingBox* out_bb = (VPIKLTTrackedBoundingBox*)outData.buffer.aos.data;
    int total = *outData.buffer.aos.sizePointer;
    int valid = 0;
    for (int i = 0; i < total; i++)
        if (out_bb[i].trackingStatus == 0) valid++;
    vpiArrayUnlock(c->tracking_estimates);

    c->num_tracked_points = valid;
    printf("[nv-stabilizer] KLT tracked %d/%d features\n", valid, total);
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_estimate_motion(NvStabCtx* c)
{
    if (c->num_tracked_points < 3) {
        c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
        c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;
        return PROC_STATUS_OK;
    }

    // Compute displacement from KLT results:
    //   reference box center (in prev_img_y)  →  tracked box center (in cur_img_y)
    //   = how much the camera moved this frame
    VPIArrayData refData, outData;
    CHECK_STATUS(vpiArrayLockData(c->keypoints_prev,     VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &refData));
    CHECK_STATUS(vpiArrayLockData(c->tracking_estimates, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &outData));

    VPIKLTTrackedBoundingBox* ref_bb = (VPIKLTTrackedBoundingBox*)refData.buffer.aos.data;
    VPIKLTTrackedBoundingBox* out_bb = (VPIKLTTrackedBoundingBox*)outData.buffer.aos.data;
    int n = *refData.buffer.aos.sizePointer;

    std::vector<float> dxs, dys;
    dxs.reserve(n); dys.reserve(n);

    for (int i = 0; i < n; i++) {
        if (out_bb[i].trackingStatus != 0) continue;  // feature lost

        // Center = translation + width/height*scale*0.5
        float ref_cx = ref_bb[i].bbox.xform.mat3[0][2] + ref_bb[i].bbox.width  * ref_bb[i].bbox.xform.mat3[0][0] * 0.5f;
        float ref_cy = ref_bb[i].bbox.xform.mat3[1][2] + ref_bb[i].bbox.height * ref_bb[i].bbox.xform.mat3[1][1] * 0.5f;
        float out_cx = out_bb[i].bbox.xform.mat3[0][2] + out_bb[i].bbox.width  * out_bb[i].bbox.xform.mat3[0][0] * 0.5f;
        float out_cy = out_bb[i].bbox.xform.mat3[1][2] + out_bb[i].bbox.height * out_bb[i].bbox.xform.mat3[1][1] * 0.5f;

        dxs.push_back(out_cx - ref_cx);
        dys.push_back(out_cy - ref_cy);
    }

    vpiArrayUnlock(c->tracking_estimates);
    vpiArrayUnlock(c->keypoints_prev);

    if (dxs.empty()) {
        c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
        c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;
        return PROC_STATUS_OK;
    }

    std::sort(dxs.begin(), dxs.end());
    std::sort(dys.begin(), dys.end());

    c->affine_matrix[0] = 1.0f;
    c->affine_matrix[1] = 0.0f;
    c->affine_matrix[2] = dxs[dxs.size() / 2];  // median camera tx
    c->affine_matrix[3] = 0.0f;
    c->affine_matrix[4] = 1.0f;
    c->affine_matrix[5] = dys[dys.size() / 2];  // median camera ty

    printf("[nv-stabilizer] KLT motion: dx=%.2f dy=%.2f (%d valid features)\n",
           c->affine_matrix[2], c->affine_matrix[5], (int)dxs.size());
    return PROC_STATUS_OK;
}

static void nv_stab_smooth_motion(NvStabCtx* c)
{
    // Trajectory-based smoothing — the correct approach for video stabilization:
    //
    //   1. Accumulate raw camera path:  traj[n] = traj[n-1] + delta[n]
    //   2. Low-pass the path:           smooth[n] = α·traj[n] + (1-α)·smooth[n-1]
    //   3. Correction = smooth[n] − traj[n]
    //
    // This separates intentional camera motion (slow, low-freq) from jitter
    // (fast, high-freq). A slow pan accumulates smoothly; high-freq shake
    // is suppressed. alpha=0.1 gives ~9-frame effective smoothing window.
    //
    // Note: affine_matrix[2/5] hold the RAW per-frame camera delta (set in
    //       estimate_motion), not a correction.

    const float alpha = 0.1f;  // trajectory smoothing (smaller = smoother / more lag)

    c->trajectory[0] += c->affine_matrix[2];  // accumulate raw camera tx
    c->trajectory[1] += c->affine_matrix[5];  // accumulate raw camera ty

    c->smoothed_trajectory[0] = alpha * c->trajectory[0] + (1.0f - alpha) * c->smoothed_trajectory[0];
    c->smoothed_trajectory[1] = alpha * c->trajectory[1] + (1.0f - alpha) * c->smoothed_trajectory[1];

    // Correction = smooth path − raw path  (negative = camera ahead, pull back)
    float corr_x = c->smoothed_trajectory[0] - c->trajectory[0];
    float corr_y = c->smoothed_trajectory[1] - c->trajectory[1];

    // Clamp correction to 90% of crop margin so we never exceed the safe border
    const int max_shift_x = (c->width  * 9) / 100;
    const int max_shift_y = (c->height * 9) / 100;
    corr_x = std::max(-(float)max_shift_x, std::min((float)max_shift_x, corr_x));
    corr_y = std::max(-(float)max_shift_y, std::min((float)max_shift_y, corr_y));

    c->smoothed_affine[0] = 1.0f; c->smoothed_affine[1] = 0.0f; c->smoothed_affine[2] = corr_x;
    c->smoothed_affine[3] = 0.0f; c->smoothed_affine[4] = 1.0f; c->smoothed_affine[5] = corr_y;

    // printf("[nv-stabilizer] traj=(%.1f,%.1f) smooth=(%.1f,%.1f) corr=(%.1f,%.1f)\n",
    //        c->trajectory[0], c->trajectory[1],
    //        c->smoothed_trajectory[0], c->smoothed_trajectory[1],
    //        corr_x, corr_y);
}

static ProcStatus nv_stab_apply_stabilization(NvStabCtx* c, VP_Frame* output)
{
    //printf("[nv-stabilizer] Applying stabilization to output frame\n");
    if (!output || !output->data) {
        //printf("[nv-stabilizer] Invalid output frame\n");
        return PROC_STATUS_ERR_GENERAL;
    }

    //printf("[nv-stabilizer] Applying stabilization to frame %lu\n", c->frame_count);
    int shift_x = (int)c->smoothed_affine[2];
    int shift_y = (int)c->smoothed_affine[5];

    if (shift_x == 0 && shift_y == 0) {
        //printf("[nv-stabilizer] No motion to compensate\n");
        return PROC_STATUS_OK;
    }

    // printf("[nv-stabilizer] Applying stabilization: shift_x=%d shift_y=%d\n", shift_x, shift_y);

    // Lock VPI images for CPU access
    VPIImageData cur_data, out_data;
    memset(&cur_data, 0, sizeof(cur_data));
    memset(&out_data, 0, sizeof(out_data));

    //printf("[nv-stabilizer] Locking current and output images\n");
    if (vpiImageLockData(c->cur_img_y, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &cur_data) != VPI_SUCCESS) {
        //printf("[nv-stabilizer] Failed to lock current image\n");
        return PROC_STATUS_OK;
    }
    //printf("[nv-stabilizer] Current image locked\n");

    if (vpiImageLockData(c->out_img_y, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &out_data) != VPI_SUCCESS) {
        vpiImageUnlock(c->cur_img_y);
        //printf("[nv-stabilizer] Failed to lock output image\n");
        return PROC_STATUS_OK;
    }
    //printf("[nv-stabilizer] Output image locked\n");

    const uint8_t* src_buf = (const uint8_t*)cur_data.buffer.pitch.planes[0].data;
    uint8_t* dst_buf = (uint8_t*)out_data.buffer.pitch.planes[0].data;
    int src_stride = cur_data.buffer.pitch.planes[0].pitchBytes;
    int dst_stride = out_data.buffer.pitch.planes[0].pitchBytes;
    int width = c->width;
    int height = c->height;

    // Crop margins: 10% border on each side to ensure no black borders appear
    const int crop_margin_x = width * 10 / 100;
    const int crop_margin_y = height * 10 / 100;

    // Cropped region dimensions (80% of original)
    const int crop_width = width - 2 * crop_margin_x;
    const int crop_height = height - 2 * crop_margin_y;

    // Apply crop, stabilize, then upscale back to original resolution
    // This eliminates black borders by only using the safe center region
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Map output pixel (x,y) to position in cropped region
            // Scale from full resolution to cropped resolution
            float crop_x_f = crop_margin_x + (x * crop_width) / (float)width;
            float crop_y_f = crop_margin_y + (y * crop_height) / (float)height;

            // Apply stabilization shift to the cropped coordinates
            float src_x_f = crop_x_f + shift_x;
            float src_y_f = crop_y_f + shift_y;

            // Bilinear interpolation for smooth upscaling
            int src_x0 = (int)src_x_f;
            int src_y0 = (int)src_y_f;
            int src_x1 = src_x0 + 1;
            int src_y1 = src_y0 + 1;

            float fx = src_x_f - src_x0;
            float fy = src_y_f - src_y0;

            // Clamp coordinates to valid range
            src_x0 = std::max(0, std::min(width - 1, src_x0));
            src_x1 = std::max(0, std::min(width - 1, src_x1));
            src_y0 = std::max(0, std::min(height - 1, src_y0));
            src_y1 = std::max(0, std::min(height - 1, src_y1));

            // Get 4 neighboring pixels
            uint8_t p00 = src_buf[src_y0 * src_stride + src_x0];
            uint8_t p10 = src_buf[src_y0 * src_stride + src_x1];
            uint8_t p01 = src_buf[src_y1 * src_stride + src_x0];
            uint8_t p11 = src_buf[src_y1 * src_stride + src_x1];

            // Bilinear interpolation
            float p0 = p00 * (1.0f - fx) + p10 * fx;
            float p1 = p01 * (1.0f - fx) + p11 * fx;
            float result = p0 * (1.0f - fy) + p1 * fy;

            dst_buf[y * dst_stride + x] = (uint8_t)(result + 0.5f);
        }
    }

    //printf("[nv-stabilizer] Stabilization applied to Y plane locally\n");
    vpiImageUnlock(c->out_img_y);
    vpiImageUnlock(c->cur_img_y);

    // Copy stabilized Y plane to output buffer
    VPIImageData stabilized_data;
    if (vpiImageLockData(c->out_img_y, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &stabilized_data) == VPI_SUCCESS) {
        const uint8_t* stabilized_src = (const uint8_t*)stabilized_data.buffer.pitch.planes[0].data;
        if (!output->data) {
            vpiImageUnlock(c->out_img_y);
            return PROC_STATUS_ERR_GENERAL;
        }
        uint8_t* output_dst = (uint8_t*)output->data;
        int y_size = width * height;
        memcpy(output_dst, stabilized_src, y_size);
        vpiImageUnlock(c->out_img_y);
    }

    // UV plane: grayscale (B&W) video has no chroma — the UV plane is entirely
    // neutral (0x80 / 128).  Shifting neutral values is a no-op, so we skip
    // the UV correction entirely to avoid a per-frame heap allocation.
    // If this is ever used with colour content, re-enable the UV shift block.

    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_process(void* vctx, VP_Frame* input)
{
    NvStabCtx* c = (NvStabCtx*)vctx;
    if (!c || !input)
        return PROC_STATUS_ERR_GENERAL;

    c->frame_count++;

    ProcStatus st;

    // 1. Prepare frame: copy Y and UV from INPUT to VPI-owned images
    st = nv_stab_prepare_frame(c, input);
    if (st != PROC_STATUS_OK) return st;

    // 2. Detect or track features
    if (c->frame_count == 1 || c->redetect_counter >= 30 || c->num_tracked_points < 20) {
        //printf("[nv-stabilizer] === Calling detect_features ===\n");
        st = nv_stab_detect_features(c);
        if (st != PROC_STATUS_OK) return st;

        // Read the actual number of detected features
        VPIArrayData kpData;
        CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &kpData));
        c->num_tracked_points = *kpData.buffer.aos.sizePointer;
        vpiArrayUnlock(c->keypoints_cur);

        c->redetect_counter = 0;
    }

    st = nv_stab_track_features(c);
    if (st != PROC_STATUS_OK) return st;
    c->redetect_counter++;

    // 3. Estimate affine motion
    st = nv_stab_estimate_motion(c);
    if (st != PROC_STATUS_OK) return st;

    // 4. Smooth motion trajectory
    nv_stab_smooth_motion(c);

    // 5. Apply stabilization
    st = nv_stab_apply_stabilization(c, input);
    if (st != PROC_STATUS_OK) return st;

    // 6. Update for next frame

    // Swap images: cur_img_y becomes the new template (prev_img_y) for next frame
    VPIImage tmp_y = c->prev_img_y;
    c->prev_img_y = c->cur_img_y;
    c->cur_img_y = tmp_y;

    VPIImage tmp_uv = c->prev_img_uv;
    c->prev_img_uv = c->cur_img_uv;
    c->cur_img_uv = tmp_uv;

    // KLT array rotation:
    //   tracking_estimates now holds feature positions inside what just became prev_img_y.
    //   Promote them to keypoints_prev (the reference for the next tracking call).
    //   Also copy them into tracked_features as the initial position prediction.
    VPIArray tmp_kp = c->keypoints_prev;
    c->keypoints_prev = c->tracking_estimates;
    c->tracking_estimates = tmp_kp;

    {
        VPIArrayData srcData, dstData;
        CHECK_STATUS(vpiArrayLockData(c->keypoints_prev,   VPI_LOCK_READ,  VPI_ARRAY_BUFFER_HOST_AOS, &srcData));
        CHECK_STATUS(vpiArrayLockData(c->tracked_features, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &dstData));
        int32_t n = *srcData.buffer.aos.sizePointer;
        memcpy(dstData.buffer.aos.data, srcData.buffer.aos.data, n * srcData.buffer.aos.strideBytes);
        *dstData.buffer.aos.sizePointer = n;
        // Predictions should not force template re-extraction (templateStatus = 0)
        VPIKLTTrackedBoundingBox* pred = (VPIKLTTrackedBoundingBox*)dstData.buffer.aos.data;
        for (int i = 0; i < n; i++) pred[i].templateStatus = 0;
        vpiArrayUnlock(c->tracked_features);
        vpiArrayUnlock(c->keypoints_prev);
    }

    return PROC_STATUS_OK;
}

static void nv_stab_destroy(void* vctx)
{
    NvStabCtx* c = (NvStabCtx*)vctx;
    if (!c) return;

    //printf("[nv-stabilizer] Destroy: processed %lu frames\n", c->frame_count);

    // Destroy Y plane images
    if (c->cur_img_y) vpiImageDestroy(c->cur_img_y);
    if (c->prev_img_y) vpiImageDestroy(c->prev_img_y);
    if (c->out_img_y) vpiImageDestroy(c->out_img_y);

    // Destroy UV plane images
    if (c->cur_img_uv) vpiImageDestroy(c->cur_img_uv);
    if (c->prev_img_uv) vpiImageDestroy(c->prev_img_uv);
    if (c->out_img_uv) vpiImageDestroy(c->out_img_uv);

    // Destroy Harris/KLT resources
    if (c->harris_payload) vpiPayloadDestroy(c->harris_payload);
    if (c->klt_payload) vpiPayloadDestroy(c->klt_payload);
    if (c->keypoints_cur) vpiArrayDestroy(c->keypoints_cur);
    if (c->harris_scores) vpiArrayDestroy(c->harris_scores);
    if (c->keypoints_prev) vpiArrayDestroy(c->keypoints_prev);
    if (c->tracked_features) vpiArrayDestroy(c->tracked_features);
    if (c->tracking_estimates) vpiArrayDestroy(c->tracking_estimates);

    if (c->vpi_stream) vpiStreamDestroy(c->vpi_stream);
    if (c->cfg_path_seen) free((void*)c->cfg_path_seen);

    free(c);
}

ProcStatus proc_register(ProcessorAPI* api)
{
    if (!api)
    return PROC_STATUS_ERR_GENERAL;

    api->init   = nv_stab_init;
    api->process = nv_stab_process;
    api->destroy = nv_stab_destroy;

    return PROC_STATUS_OK;
}
