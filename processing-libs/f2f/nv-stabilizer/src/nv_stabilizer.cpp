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
    VPIArray keypoints_cur;
    VPIArray keypoints_prev;
    VPIHarrisCornerDetectorParams harris_params;

    // KLT feature tracking
    VPIPayload klt_payload;
    VPIArray tracked_features;
    VPIArray tracking_estimates;
    VPIKLTFeatureTrackerParams klt_params;

    // Motion estimation
    float affine_matrix[6];        // [a b tx c d ty]
    float smoothed_affine[6];      // Temporally filtered

    // Motion history for smoothing
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
        if (c->keypoints_prev) vpiArrayDestroy(c->keypoints_prev);
        if (c->tracked_features) vpiArrayDestroy(c->tracked_features);
        if (c->tracking_estimates) vpiArrayDestroy(c->tracking_estimates);

        // Create keypoint arrays (on processing thread for thread safety)
        //printf("[nv-stabilizer] Creating VPI arrays\n");
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c->keypoints_cur));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c->keypoints_prev));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KEYPOINT_F32, 0, &c->tracked_features));
        CHECK_STATUS(vpiArrayCreate(500, VPI_ARRAY_TYPE_KLT_TRACKED_BOUNDING_BOX, 0, &c->tracking_estimates));

        // Create Harris corner detector
        //printf("[nv-stabilizer] Creating Harris detector %dx%d\n", w, h);
        VPIStatus harris_st = vpiCreateHarrisCornerDetector(VPI_BACKEND_CUDA, w, h, &c->harris_payload);
        if (harris_st != VPI_SUCCESS) {
            //printf("[nv-stabilizer] Harris creation FAILED: %s\n", vpiStatusGetName(harris_st));
            c->harris_payload = NULL;
        } else {
            //printf("[nv-stabilizer] Harris created: %p\n", c->harris_payload);
        }

        // Create KLT feature tracker
        CHECK_STATUS(vpiCreateKLTFeatureTracker(VPI_BACKEND_CUDA, w, h,
                                                 VPI_IMAGE_FORMAT_Y8, 0, &c->klt_payload));

        c->has_prev_features = false;
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
    // Grid-based keypoint generation
    //printf("[nv-stabilizer] detect_features: c=%p, keypoints_cur=%p\n", c, c->keypoints_cur);

    VPIArrayData arrInit;
    //printf("[nv-stabilizer] About to lock keypoints_cur array...\n");

    CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &arrInit));

    //printf("[nv-stabilizer] Successfully locked array\n");

    VPIKeypointF32* kpts = (VPIKeypointF32*)arrInit.buffer.aos.data;
    int num_kpts = 0;
    const int grid_spacing = 100;

    //printf("[nv-stabilizer] Starting keypoint generation loop\n");

    for (int y = 50; y < c->height && num_kpts < 500; y += grid_spacing) {
        for (int x = 50; x < c->width && num_kpts < 500; x += grid_spacing) {
            kpts[num_kpts].x = (float)x;
            kpts[num_kpts].y = (float)y;
            num_kpts++;
        }
    }

    //printf("[nv-stabilizer] Generated %d keypoints, setting size pointer\n", num_kpts);

    *arrInit.buffer.aos.sizePointer = num_kpts;

    //printf("[nv-stabilizer] Unlocking array\n");

    vpiArrayUnlock(c->keypoints_cur);

    //printf("[nv-stabilizer] Generated %d grid-based keypoints\n", num_kpts);
    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_track_features(NvStabCtx *c)
{
    if (!c->has_prev_features) {
        // First frame: just copy keypoints to prev and return
        VPIArrayData curData, prevData;
        CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &curData));
        CHECK_STATUS(vpiArrayLockData(c->keypoints_prev, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &prevData));

        int32_t numElements = *curData.buffer.aos.sizePointer;
        size_t copySize = numElements * curData.buffer.aos.strideBytes;
        memcpy(prevData.buffer.aos.data, curData.buffer.aos.data, copySize);
        *prevData.buffer.aos.sizePointer = numElements;

        vpiArrayUnlock(c->keypoints_prev);
        vpiArrayUnlock(c->keypoints_cur);

        c->has_prev_features = true;
        c->num_tracked_points = numElements;
        //printf("[nv-stabilizer] First frame: stored %d grid keypoints\n", numElements);
        return PROC_STATUS_OK;
    }

    // For grid-based points, simply re-generate and compute displacement
    // Copy current to previous for motion estimation
    VPIArrayData curData, prevData;
    CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &curData));
    CHECK_STATUS(vpiArrayLockData(c->keypoints_prev, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &prevData));

    int32_t numElements = *curData.buffer.aos.sizePointer;
    size_t copySize = numElements * curData.buffer.aos.strideBytes;
    memcpy(prevData.buffer.aos.data, curData.buffer.aos.data, copySize);
    *prevData.buffer.aos.sizePointer = numElements;

    vpiArrayUnlock(c->keypoints_prev);
    vpiArrayUnlock(c->keypoints_cur);

    c->num_tracked_points = numElements;
    //printf("[nv-stabilizer] Prepared %d grid points for motion estimation\n", numElements);

    return PROC_STATUS_OK;
}

static ProcStatus nv_stab_estimate_motion(NvStabCtx* c)
{
    if (c->num_tracked_points < 3) {
        // Not enough points for affine estimation, use identity
        c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
        c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;
        //printf("[nv-stabilizer] Insufficient points, using identity motion\n");
        return PROC_STATUS_OK;
    }

    // Read current and previous frame Y data directly for simple registration
    // Use template matching on grid regions for robust motion estimation
    VPIImageData cur_img_data, prev_img_data;
    CHECK_STATUS(vpiImageLockData(c->cur_img_y, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &cur_img_data));
    CHECK_STATUS(vpiImageLockData(c->prev_img_y, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &prev_img_data));

    uint8_t* cur_buf = (uint8_t*)cur_img_data.buffer.pitch.planes[0].data;
    uint8_t* prev_buf = (uint8_t*)prev_img_data.buffer.pitch.planes[0].data;
    int pitch = cur_img_data.buffer.pitch.planes[0].pitchBytes;
    int width = c->width;
    int height = c->height;

    // Read keypoints to get grid positions
    VPIArrayData kp_data;
    CHECK_STATUS(vpiArrayLockData(c->keypoints_cur, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS, &kp_data));
    VPIKeypointF32* kpts = (VPIKeypointF32*)kp_data.buffer.aos.data;
    int num_kpts = *kp_data.buffer.aos.sizePointer;

    // Simple block matching at grid points
    std::vector<float> dxs, dys;
    const int template_size = 15;  // 15x15 template
    const int search_range = 30;   // Search ±30 pixels

    for (int k = 0; k < std::min(num_kpts, 50); k++) {
        int x = (int)kpts[k].x;
        int y = (int)kpts[k].y;

        // Bounds check for template extraction
        if (x - template_size/2 < 0 || x + template_size/2 >= width ||
            y - template_size/2 < 0 || y + template_size/2 >= height) {
            continue;
        }

        // Extract template from previous frame
        int best_dx = 0, best_dy = 0;
        int best_error = INT_MAX;

        for (int dy = -search_range; dy <= search_range; dy++) {
            for (int dx = -search_range; dx <= search_range; dx++) {
                int nx = x + dx;
                int ny = y + dy;

                if (nx - template_size/2 < 0 || nx + template_size/2 >= width ||
                    ny - template_size/2 < 0 || ny + template_size/2 >= height) {
                    continue;
                }

                // Compute SAD (Sum of Absolute Differences)
                int error = 0;
                for (int ty = -template_size/2; ty <= template_size/2; ty++) {
                    for (int tx = -template_size/2; tx <= template_size/2; tx++) {
                        int prev_idx = (y + ty) * pitch + (x + tx);
                        int cur_idx = (ny + ty) * pitch + (nx + tx);
                        error += abs((int)prev_buf[prev_idx] - (int)cur_buf[cur_idx]);
                    }
                }

                if (error < best_error) {
                    best_error = error;
                    best_dx = dx;
                    best_dy = dy;
                }
            }
        }

        dxs.push_back((float)best_dx);
        dys.push_back((float)best_dy);
    }

    vpiArrayUnlock(c->keypoints_cur);
    vpiImageUnlock(c->prev_img_y);
    vpiImageUnlock(c->cur_img_y);

    if (dxs.empty()) {
        //printf("[nv-stabilizer] No valid block matches, using identity\n");
        c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
        c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;
        return PROC_STATUS_OK;
    }

    // Median filter to get robust motion estimate
    std::sort(dxs.begin(), dxs.end());
    std::sort(dys.begin(), dys.end());
    float med_dx = dxs[dxs.size() / 2];
    float med_dy = dys[dys.size() / 2];

    // Build affine transform (translation only)
    // Inverse for stabilization: move frame opposite to camera motion
    c->affine_matrix[0] = 1.0f;
    c->affine_matrix[1] = 0.0f;
    c->affine_matrix[2] = -med_dx;  // tx (inverse)
    c->affine_matrix[3] = 0.0f;
    c->affine_matrix[4] = 1.0f;
    c->affine_matrix[5] = -med_dy;  // ty (inverse)

    //printf("[nv-stabilizer] Block match motion: dx=%.2f dy=%.2f (from %d matches)\n", med_dx, med_dy, (int)dxs.size());

    return PROC_STATUS_OK;
}

static void nv_stab_smooth_motion(NvStabCtx* c)
{
    // Add current motion to history
    memcpy(c->motion_history[c->history_index], c->affine_matrix, sizeof(c->affine_matrix));
    c->history_index = (c->history_index + 1) % MOTION_HISTORY_SIZE;
    if (c->history_index == 0) c->history_full = true;

    // Compute exponential moving average
    const float alpha = 0.3f;  // Smoothing factor (0 = max smooth, 1 = no smooth)

    if (!c->history_full && c->history_index == 0) {
        // First frame, no smoothing yet
        memcpy(c->smoothed_affine, c->affine_matrix, sizeof(c->affine_matrix));
    } else {
        // Smooth each component
        for (int i = 0; i < 6; i++) {
            c->smoothed_affine[i] = alpha * c->affine_matrix[i] +
                                   (1.0f - alpha) * c->smoothed_affine[i];
        }
    }

    //printf("[nv-stabilizer] Smoothed: tx=%.2f ty=%.2f\n", c->smoothed_affine[2], c->smoothed_affine[5]);
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

    printf("[nv-stabilizer] Applying stabilization: shift_x=%d shift_y=%d\n", shift_x, shift_y);

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

    // Apply frame shifting: compensate for detected motion
    // To stabilize: shift frame in OPPOSITE direction of detected motion
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Source coordinates after inverse shift
            int src_x = x - shift_x;
            int src_y = y - shift_y;

            if (src_x < 0 || src_x >= width || src_y < 0 || src_y >= height) {
                // Out of bounds: fill with black
                dst_buf[y * dst_stride + x] = 0;
            } else {
                // In bounds: copy from source
                dst_buf[y * dst_stride + x] = src_buf[src_y * src_stride + src_x];
            }
        }
    }

    //printf("[nv-stabilizer] Stabilization applied to Y plane locally\n");
    vpiImageUnlock(c->out_img_y);
    vpiImageUnlock(c->cur_img_y);

    //printf("[nv-stabilizer] Copying stabilized Y plane to output frame\n");
    VPIImageData stabilized_data;
    if (vpiImageLockData(c->out_img_y, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &stabilized_data) == VPI_SUCCESS) {
        //printf("[nv-stabilizer] Locked stabilized image for output copy\n");
        const uint8_t* stabilized_src = (const uint8_t*)stabilized_data.buffer.pitch.planes[0].data;
        //printf("[nv-stabilizer] after stabilized_src\n");
        if(!output->data) {
            //printf("[nv-stabilizer] Output frame has no data buffer\n");
            vpiImageUnlock(c->out_img_y);
            return PROC_STATUS_ERR_GENERAL;
        }
        //printf("[nv-stabilizer] before output_dst\n");
        uint8_t* output_dst = (uint8_t*)output->data;
        int y_size = width * height;

        // Copy Y plane only (UV passes through unchanged from input)
        //printf("[nv-stabilizer] Copying bytes of Y plane to output\n");
        memcpy(output_dst, stabilized_src, y_size);

        vpiImageUnlock(c->out_img_y);
        //printf("[nv-stabilizer] Stabilization applied to Y plane\n");
    }
    //printf("[nv-stabilizer] Stabilization process complete for frame %lu\n", c->frame_count);
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

    // 6. Update for next frame: swap images
    VPIImage tmp_y = c->prev_img_y;
    c->prev_img_y = c->cur_img_y;
    c->cur_img_y = tmp_y;

    VPIImage tmp_uv = c->prev_img_uv;
    c->prev_img_uv = c->cur_img_uv;
    c->cur_img_uv = tmp_uv;

    // Swap keypoints
    VPIArray tmp_kp = c->keypoints_prev;
    c->keypoints_prev = c->keypoints_cur;
    c->keypoints_cur = tmp_kp;

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
