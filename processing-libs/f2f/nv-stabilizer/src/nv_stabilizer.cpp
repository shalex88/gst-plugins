#include <sstream>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cctype>

#include <yaml-cpp/yaml.h>

#include <vpi/VPI.h>
#include <vpi/algo/ConvertImageFormat.h>
#include <vpi/algo/OpticalFlowDense.h>
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



typedef struct {
    int ofa_grid_size;  // OFA block grid (larger = faster, less detail).
    VPIOpticalFlowQuality ofa_quality;  // OFA quality/power trade-off.
    int ofa_sample_step;  // Subsample MV grid when fitting motion.

    float raw_translation_blend_ls;  // Blend weight for LS translation estimate.
    float raw_translation_blend_med;  // Blend weight for median translation.
    float raw_rotation_scale;  // Damp raw rotation to reduce jitter.

    float filter_beta;  // Low-pass on per-frame motion.
    float traj_alpha;  // Trajectory smoothing factor.
    float correction_gain;  // Strength of correction vs. raw path.

    float correction_deadzone_px;  // Ignore tiny subpixel corrections.
    float correction_max_rot;  // Max correction rotation (radians).

    float zoom_min_scale;  // Minimum zoom-in allowed.
    float zoom_smoothing;  // Zoom easing factor.
    float zoom_margin_pad_px;  // Extra margin to avoid borders.
} StabParams;

static VPIOpticalFlowQuality parse_ofa_quality(const YAML::Node &node, VPIOpticalFlowQuality fallback)
{
    if (!node) {
        return fallback;
    }

    try {
        if (node.IsScalar()) {
            std::string value = node.as<std::string>();
            for (char &ch : value) {
                ch = (char)std::tolower((unsigned char)ch);
            }
            if (value == "low") {
                return VPI_OPTICAL_FLOW_QUALITY_LOW;
            }
            if (value == "medium") {
                return VPI_OPTICAL_FLOW_QUALITY_MEDIUM;
            }
            if (value == "high") {
                return VPI_OPTICAL_FLOW_QUALITY_HIGH;
            }
            return (VPIOpticalFlowQuality)node.as<int>();
        }
    } catch (const std::exception &) {
    }

    return fallback;
}

// static StabParams default_params()
// {
//     StabParams p{};
//     p.ofa_grid_size = 4;
//     p.ofa_quality = VPI_OPTICAL_FLOW_QUALITY_MEDIUM;
//     p.ofa_sample_step = 2;

//     p.raw_translation_blend_ls = 0.6f;
//     p.raw_translation_blend_med = 0.4f;
//     p.raw_rotation_scale = 0.6f;

//     p.filter_beta = 0.25f;
//     p.traj_alpha = 0.05f;
//     p.correction_gain = 1.3f;

//     p.correction_deadzone_px = 0.25f;
//     p.correction_max_rot = 0.15f;

//     p.zoom_min_scale = 0.90f;
//     p.zoom_smoothing = 0.05f;
//     p.zoom_margin_pad_px = 8.0f;
//     return p;
// }

static void load_params_from_yaml(StabParams &params, const char *path)
{
    if (!path || !*path) {
        return;
    }

    try {
        YAML::Node cfg = YAML::LoadFile(path);
        const YAML::Node stab = cfg["stabilizer"].IsDefined() ? cfg["stabilizer"] : cfg;
        YAML::Node params_node = stab;
        if (stab["presets"] && stab["selected_preset"]) {
            std::string preset = stab["selected_preset"].as<std::string>();
            if (stab["presets"][preset]) {
                params_node = stab["presets"][preset];
            }
        }

        if (params_node["ofa_grid_size"]) params.ofa_grid_size = params_node["ofa_grid_size"].as<int>();
        if (params_node["ofa_sample_step"]) params.ofa_sample_step = params_node["ofa_sample_step"].as<int>();
        params.ofa_quality = parse_ofa_quality(params_node["ofa_quality"], params.ofa_quality);

        if (params_node["raw_translation_blend_ls"]) params.raw_translation_blend_ls = params_node["raw_translation_blend_ls"].as<float>();
        if (params_node["raw_translation_blend_med"]) params.raw_translation_blend_med = params_node["raw_translation_blend_med"].as<float>();
        if (params_node["raw_rotation_scale"]) params.raw_rotation_scale = params_node["raw_rotation_scale"].as<float>();

        if (params_node["filter_beta"]) params.filter_beta = params_node["filter_beta"].as<float>();
        if (params_node["traj_alpha"]) params.traj_alpha = params_node["traj_alpha"].as<float>();
        if (params_node["correction_gain"]) params.correction_gain = params_node["correction_gain"].as<float>();

        if (params_node["correction_deadzone_px"]) params.correction_deadzone_px = params_node["correction_deadzone_px"].as<float>();
        if (params_node["correction_max_rot"]) params.correction_max_rot = params_node["correction_max_rot"].as<float>();

        if (params_node["zoom_min_scale"]) params.zoom_min_scale = params_node["zoom_min_scale"].as<float>();
        if (params_node["zoom_smoothing"]) params.zoom_smoothing = params_node["zoom_smoothing"].as<float>();
        if (params_node["zoom_margin_pad_px"]) params.zoom_margin_pad_px = params_node["zoom_margin_pad_px"].as<float>();
    } catch (const std::exception &e) {
        std::cerr << "[nv-stabilizer] config load failed: " << e.what() << "\n";
    }
}

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

    VPIImage cur_img_y_bl;
    VPIImage prev_img_y_bl;

    VPIImage mv_img_bl;
    VPIImage mv_img_pl;

    VPIPayload ofa_payload;
    int of_grid;
    int mv_width;
    int mv_height;

    VPIImage cur_img_uv;
    VPIImage prev_img_uv;
    VPIImage out_img_uv;

    // Harris corner detection
    VPIPayload harris_payload;
    VPIArray keypoints_cur;        // Harris output keypoints (VPI_ARRAY_TYPE_KEYPOINT_F32)
    VPIArray harris_scores;        // Harris output scores    (VPI_ARRAY_TYPE_U32)
    VPIHarrisCornerDetectorParams harris_params;

    // KLT feature tracking
    // keypoints_prev    — reference boxes in the template frame   (KLT_TRACKED_BOUNDING_BOX)
    // tracked_features  — initial position predictions             (HOMOGRAPHY_TRANSFORM_2D)
    // tracking_estimates— KLT output: tracked boxes in cur frame   (KLT_TRACKED_BOUNDING_BOX)
    VPIPayload klt_payload;
    VPIArray keypoints_prev;
    VPIArray tracked_features;
    VPIArray tracking_estimates;
    VPIArray tracking_transforms;
    VPIKLTFeatureTrackerParams klt_params;

    // Motion estimation
    float affine_matrix[6];        // [a b tx c d ty] — raw per-frame camera delta
    float smoothed_affine[6];      // Warp to apply this frame (correction)
    float raw_rotation;
    float trajectory_rot;
    float smoothed_trajectory_rot;
    float corr_tx;
    float corr_ty;
    float corr_rot;
    float filt_tx;
    float filt_ty;
    float filt_rot;
    float zoom_scale;

    // Trajectory-based smoothing
    // traj = accumulated raw camera path; smooth_traj = low-pass of traj
    // correction = smooth_traj - traj  (how far to shift to follow smooth path)
    float trajectory[2];           // accumulated (tx, ty)
    float smoothed_trajectory[2];  // EMA of trajectory

    // State
    bool has_prev_features;
    bool has_prev_of;
    int num_tracked_points;
    int redetect_counter;

    StabParams params;

} NvStabCtx;

static void copy_plane_host_to_vpi(uint8_t *dst, int dst_pitch,
                                   const uint8_t *src, int src_pitch,
                                   int width, int height)
{
    for (int y = 0; y < height; ++y) {
        memcpy(dst + y * dst_pitch, src + y * src_pitch, width);
    }
}

static bool solve_linear_6x6(double a[6][6], double b[6], double x[6])
{
    for (int i = 0; i < 6; ++i) {
        int pivot = i;
        double max_val = std::fabs(a[i][i]);
        for (int r = i + 1; r < 6; ++r) {
            double v = std::fabs(a[r][i]);
            if (v > max_val) {
                max_val = v;
                pivot = r;
            }
        }

        if (max_val < 1e-9) {
            return false;
        }

        if (pivot != i) {
            for (int c = i; c < 6; ++c) {
                std::swap(a[i][c], a[pivot][c]);
            }
            std::swap(b[i], b[pivot]);
        }

        double inv = 1.0 / a[i][i];
        for (int c = i; c < 6; ++c) {
            a[i][c] *= inv;
        }
        b[i] *= inv;

        for (int r = 0; r < 6; ++r) {
            if (r == i) continue;
            double f = a[r][i];
            if (std::fabs(f) < 1e-12) continue;
            for (int c = i; c < 6; ++c) {
                a[r][c] -= f * a[i][c];
            }
            b[r] -= f * b[i];
        }
    }

    for (int i = 0; i < 6; ++i) {
        x[i] = b[i];
    }

    return true;
}

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
    c->has_prev_of = false;
    c->num_tracked_points = 0;
    c->redetect_counter = 0;

    // VPI stream will be created lazily on first frame (thread safety)
    c->vpi_stream = NULL;

    // Harris/KLT unused in optical-flow pipeline

    // Initialize identity affine transform
    c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] = 0.0f;
    c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] = 0.0f;

    memcpy(c->smoothed_affine, c->affine_matrix, sizeof(c->affine_matrix));

    c->trajectory[0]          = 0.0f; c->trajectory[1]          = 0.0f;
    c->smoothed_trajectory[0] = 0.0f; c->smoothed_trajectory[1] = 0.0f;
    c->trajectory_rot = 0.0f;
    c->smoothed_trajectory_rot = 0.0f;
    c->corr_tx = 0.0f;
    c->corr_ty = 0.0f;
    c->corr_rot = 0.0f;
    c->filt_tx = 0.0f;
    c->filt_ty = 0.0f;
    c->filt_rot = 0.0f;
    c->zoom_scale = 1.0f;

    // c->params = default_params();
    load_params_from_yaml(c->params, config_path);
    std::cout << "[nv-stabilizer] config: path=" << (config_path ? config_path : "(none)")
              << " ofa_grid_size=" << c->params.ofa_grid_size
              << " ofa_quality=" << (int)c->params.ofa_quality
              << " correction_gain=" << c->params.correction_gain
              << " zoom_min_scale=" << c->params.zoom_min_scale
              << "\n";

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

        if (c->cur_img_y_bl) vpiImageDestroy(c->cur_img_y_bl);
        if (c->prev_img_y_bl) vpiImageDestroy(c->prev_img_y_bl);
        if (c->mv_img_bl) vpiImageDestroy(c->mv_img_bl);
        if (c->mv_img_pl) vpiImageDestroy(c->mv_img_pl);

        // Destroy old UV plane images
        if (c->cur_img_uv) vpiImageDestroy(c->cur_img_uv);
        if (c->prev_img_uv) vpiImageDestroy(c->prev_img_uv);
        if (c->out_img_uv) vpiImageDestroy(c->out_img_uv);

        // Create Y plane images
        //printf("[nv-stabilizer] Creating Y images %dx%d\n", w, h);
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->cur_img_y));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->prev_img_y));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER, 0, &c->out_img_y));
        //printf("[nv-stabilizer] Y images created: cur=%p prev=%p out=%p\n", c->cur_img_y, c->prev_img_y, c->out_img_y);

        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &c->cur_img_y_bl));
        CHECK_STATUS(vpiImageCreate(w, h, VPI_IMAGE_FORMAT_Y8_ER_BL, 0, &c->prev_img_y_bl));

        c->of_grid = c->params.ofa_grid_size;
        c->mv_width = (w + c->of_grid - 1) / c->of_grid;
        c->mv_height = (h + c->of_grid - 1) / c->of_grid;
        CHECK_STATUS(vpiImageCreate(c->mv_width, c->mv_height, VPI_IMAGE_FORMAT_2S16_BL, 0, &c->mv_img_bl));
        CHECK_STATUS(vpiImageCreate(c->mv_width, c->mv_height, VPI_IMAGE_FORMAT_2S16, 0, &c->mv_img_pl));

        // UV plane images disabled - not used in current stabilization implementation
        //printf("[nv-stabilizer] Skipping UV image creation (Y-plane only stabilization)\n");
        c->cur_img_uv = NULL;
        c->prev_img_uv = NULL;
        c->out_img_uv = NULL;

        if (c->ofa_payload) vpiPayloadDestroy(c->ofa_payload);
        c->ofa_payload = NULL;
        int32_t grid = c->of_grid;
        VPIStatus ofa_st = vpiCreateOpticalFlowDense(VPI_BACKEND_OFA, w, h,
                                                     VPI_IMAGE_FORMAT_Y8_ER_BL, &grid, 1,
                     c->params.ofa_quality, &c->ofa_payload);
        if (ofa_st != VPI_SUCCESS) {
            c->ofa_payload = NULL;
            return PROC_STATUS_ERR_GENERAL;
        }

        c->has_prev_features = false;
        c->has_prev_of = false;

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
    {
        VPIImageData imgData;
        CHECK_STATUS(vpiImageLockData(c->cur_img_y, VPI_LOCK_WRITE, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &imgData));

        const uint8_t* src = (const uint8_t*)frame->data;
        uint8_t* dst = (uint8_t*)imgData.buffer.pitch.planes[0].data;
        const int dst_pitch = imgData.buffer.pitch.planes[0].pitchBytes;
        copy_plane_host_to_vpi(dst, dst_pitch, src, frame->stride, frame->width, frame->height);

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

    // Convert keypoints → KLT reference boxes + identity homography predictions.
    // keypoints_prev   = reference boxes for the KLT template (cur_img_y; after the
    //                    per-frame swap this will be prev_img_y which the tracker reads).
    // tracked_features = identity predictions (no prior motion known).
    const float BOX = 21.0f;  // 21×21 pixel patch
    const float HALF = BOX / 2.0f;

    VPIArrayData kpData, refData, predData;
    CHECK_STATUS(vpiArrayLockData(c->keypoints_cur,    VPI_LOCK_READ,  VPI_ARRAY_BUFFER_HOST_AOS, &kpData));
    CHECK_STATUS(vpiArrayLockData(c->keypoints_prev,   VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &refData));
    CHECK_STATUS(vpiArrayLockData(c->tracked_features, VPI_LOCK_WRITE, VPI_ARRAY_BUFFER_HOST_AOS, &predData));

    VPIKeypointF32*           kpts    = (VPIKeypointF32*)kpData.buffer.aos.data;
    VPIKLTTrackedBoundingBox* ref_bb  = (VPIKLTTrackedBoundingBox*)refData.buffer.aos.data;
    VPIHomographyTransform2D* pred_tf = (VPIHomographyTransform2D*)predData.buffer.aos.data;
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

        // Initialise reference box in template image.
        memset(&ref_bb[i].bbox.xform, 0, sizeof(ref_bb[i].bbox.xform));
        ref_bb[i].bbox.xform.mat3[0][0] = 1.0f;  // x scale
        ref_bb[i].bbox.xform.mat3[1][1] = 1.0f;  // y scale
        ref_bb[i].bbox.xform.mat3[2][2] = 1.0f;  // homogeneous
        ref_bb[i].bbox.xform.mat3[0][2] = bx;    // left
        ref_bb[i].bbox.xform.mat3[1][2] = by;    // top
        ref_bb[i].bbox.width  = BOX;
        ref_bb[i].bbox.height = BOX;
        ref_bb[i].trackingStatus = 0;  // valid

        // Initialise prediction transform as identity.
        memset(&pred_tf[i], 0, sizeof(pred_tf[i]));
        pred_tf[i].mat3[0][0] = 1.0f;
        pred_tf[i].mat3[1][1] = 1.0f;
        pred_tf[i].mat3[2][2] = 1.0f;
        ref_bb[i].templateStatus  = 1;  // force template extraction on first track
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
        c->tracking_transforms,
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
    VPIImageData mv_data;
    CHECK_STATUS(vpiImageLockData(c->mv_img_pl, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &mv_data));

    const int16_t* mv_ptr = (const int16_t*)mv_data.buffer.pitch.planes[0].data;
    const int mv_pitch = mv_data.buffer.pitch.planes[0].pitchBytes / sizeof(int16_t);

    const float max_abs_dx = c->width * 0.25f;
    const float max_abs_dy = c->height * 0.25f;

    double ata[6][6] = {};
    double atb[6] = {};
    double sum_dx = 0.0;
    double sum_dy = 0.0;
    int count = 0;

    std::vector<float> dxs;
    std::vector<float> dys;

    const int step = c->params.ofa_sample_step;
    for (int y = 0; y < c->mv_height; y += step) {
        for (int x = 0; x < c->mv_width; x += step) {
            const int idx = y * mv_pitch + (x * 2);
            float dx = mv_ptr[idx] / 32.0f;
            float dy = mv_ptr[idx + 1] / 32.0f;

            if (std::fabs(dx) > max_abs_dx || std::fabs(dy) > max_abs_dy) {
                continue;
            }

            const float px = (x + 0.5f) * c->of_grid;
            const float py = (y + 0.5f) * c->of_grid;
            const float qx = px + dx;
            const float qy = py + dy;

            const double r1[6] = {px, py, 1.0, 0.0, 0.0, 0.0};
            const double r2[6] = {0.0, 0.0, 0.0, px, py, 1.0};

            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    ata[i][j] += r1[i] * r1[j] + r2[i] * r2[j];
                }
                atb[i] += r1[i] * qx + r2[i] * qy;
            }

            sum_dx += dx;
            sum_dy += dy;
            count++;

            dxs.push_back(dx);
            dys.push_back(dy);
        }
    }

    vpiImageUnlock(c->mv_img_pl);

    if (count < 20) {
        c->affine_matrix[0] = 1.0f; c->affine_matrix[1] = 0.0f; c->affine_matrix[2] *= 0.7f;
        c->affine_matrix[3] = 0.0f; c->affine_matrix[4] = 1.0f; c->affine_matrix[5] *= 0.7f;
        return PROC_STATUS_OK;
    }

    double x[6];
    bool ok = solve_linear_6x6(ata, atb, x);
    float a = 1.0f, b = 0.0f, cterm = 0.0f, d = 1.0f, tx = 0.0f, ty = 0.0f;

    if (ok) {
        a = (float)x[0]; b = (float)x[1]; tx = (float)x[2];
        cterm = (float)x[3]; d = (float)x[4]; ty = (float)x[5];
    } else {
        tx = (float)(sum_dx / count);
        ty = (float)(sum_dy / count);
    }

    if (!dxs.empty()) {
        std::nth_element(dxs.begin(), dxs.begin() + dxs.size() / 2, dxs.end());
        std::nth_element(dys.begin(), dys.begin() + dys.size() / 2, dys.end());
        float med_dx = dxs[dxs.size() / 2];
        float med_dy = dys[dys.size() / 2];
        tx = c->params.raw_translation_blend_ls * tx + c->params.raw_translation_blend_med * med_dx;
        ty = c->params.raw_translation_blend_ls * ty + c->params.raw_translation_blend_med * med_dy;
    }

    float scale = std::sqrt(a * a + cterm * cterm);
    if (scale < 0.9f || scale > 1.1f) {
        a = 1.0f; b = 0.0f; cterm = 0.0f; d = 1.0f;
    }

    float rot = std::atan2(cterm, a);
    const float max_rot = 0.2f;
    if (std::fabs(rot) > max_rot) {
        rot = (rot > 0.0f) ? max_rot : -max_rot;
        a = std::cos(rot);
        b = -std::sin(rot);
        cterm = std::sin(rot);
        d = std::cos(rot);
    }

    rot *= c->params.raw_rotation_scale;

    tx = std::max(-max_abs_dx, std::min(max_abs_dx, tx));
    ty = std::max(-max_abs_dy, std::min(max_abs_dy, ty));

    c->affine_matrix[0] = a;
    c->affine_matrix[1] = b;
    c->affine_matrix[2] = tx;
    c->affine_matrix[3] = cterm;
    c->affine_matrix[4] = d;
    c->affine_matrix[5] = ty;

    c->raw_rotation = rot;

    printf("[nv-stabilizer] OFA motion: tx=%.2f ty=%.2f rot=%.3f (%d samples)\n",
           tx, ty, rot, count);
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

    const float alpha = c->params.traj_alpha;  // trajectory smoothing (smaller = smoother / more lag)
    const float correction_gain = c->params.correction_gain;

    const float beta = c->params.filter_beta;
    c->filt_tx = (1.0f - beta) * c->filt_tx + beta * c->affine_matrix[2];
    c->filt_ty = (1.0f - beta) * c->filt_ty + beta * c->affine_matrix[5];
    c->filt_rot = (1.0f - beta) * c->filt_rot + beta * c->raw_rotation;

    c->trajectory[0] += c->filt_tx;
    c->trajectory[1] += c->filt_ty;
    c->trajectory_rot += c->filt_rot;

    c->smoothed_trajectory[0] = alpha * c->trajectory[0] + (1.0f - alpha) * c->smoothed_trajectory[0];
    c->smoothed_trajectory[1] = alpha * c->trajectory[1] + (1.0f - alpha) * c->smoothed_trajectory[1];

    c->smoothed_trajectory_rot = alpha * c->trajectory_rot + (1.0f - alpha) * c->smoothed_trajectory_rot;

    float corr_x = (c->smoothed_trajectory[0] - c->trajectory[0]) * correction_gain;
    float corr_y = (c->smoothed_trajectory[1] - c->trajectory[1]) * correction_gain;
    float corr_r = (c->smoothed_trajectory_rot - c->trajectory_rot) * correction_gain;

    const float deadzone_px = c->params.correction_deadzone_px;
    if (std::fabs(corr_x) < deadzone_px) corr_x = 0.0f;
    if (std::fabs(corr_y) < deadzone_px) corr_y = 0.0f;

    // Clamp correction to 90% of crop margin so we never exceed the safe border
    const int max_shift_x = (c->width  * 10) / 100;
    const int max_shift_y = (c->height * 10) / 100;
    corr_x = std::max(-(float)max_shift_x, std::min((float)max_shift_x, corr_x));
    corr_y = std::max(-(float)max_shift_y, std::min((float)max_shift_y, corr_y));

    const float max_rot = c->params.correction_max_rot;
    corr_r = std::max(-max_rot, std::min(max_rot, corr_r));

    float desired_margin = std::max(std::fabs(corr_x), std::fabs(corr_y)) + c->params.zoom_margin_pad_px;
    float max_margin = std::min((float)max_shift_x, (float)max_shift_y);
    desired_margin = std::min(desired_margin, max_margin);
    float desired_scale = (c->width - 2.0f * desired_margin) / (float)c->width;
    desired_scale = std::max(c->params.zoom_min_scale, std::min(1.0f, desired_scale));
    c->zoom_scale = (1.0f - c->params.zoom_smoothing) * c->zoom_scale + c->params.zoom_smoothing * desired_scale;

    c->corr_tx = corr_x;
    c->corr_ty = corr_y;
    c->corr_rot = corr_r;

    if ((c->frame_count % 30) == 0) {
        printf("[nv-stabilizer] correction: tx=%.2f ty=%.2f rot=%.3f\n",
               corr_x, corr_y, corr_r);
    }

    // printf("[nv-stabilizer] traj=(%.1f,%.1f) smooth=(%.1f,%.1f) corr=(%.1f,%.1f)\n",
    //        c->trajectory[0], c->trajectory[1],
    //        c->smoothed_trajectory[0], c->smoothed_trajectory[1],
    //        corr_x, corr_y);
}

static ProcStatus nv_stab_apply_stabilization(NvStabCtx* c, VP_Frame* output)
{
    if (!output || !output->data) {
        return PROC_STATUS_ERR_GENERAL;
    }

    const float s = c->zoom_scale;

    const float cx = c->width * 0.5f;
    const float cy = c->height * 0.5f;

    const float cos_r = std::cos(c->corr_rot);
    const float sin_r = std::sin(c->corr_rot);

    const float a = s * cos_r;
    const float b = -s * sin_r;
    const float d = s * cos_r;
    const float cterm = s * sin_r;

    const float tx = cx + c->corr_tx - a * cx - b * cy;
    const float ty = cy + c->corr_ty - cterm * cx - d * cy;

    VPIPerspectiveTransform xform = {
        {a, b, tx},
        {cterm, d, ty},
        {0.0f, 0.0f, 1.0f}
    };

    CHECK_STATUS(vpiSubmitPerspectiveWarp(c->vpi_stream,
                                          VPI_BACKEND_CUDA,
                                          c->cur_img_y,
                                          xform,
                                          c->out_img_y,
                                          NULL,
                                          VPI_INTERP_LINEAR,
                                          VPI_BORDER_ZERO,
                                          VPI_WARP_INVERSE));
    CHECK_STATUS(vpiStreamSync(c->vpi_stream));

    // Copy stabilized Y plane to output buffer
    VPIImageData stabilized_data;
    if (vpiImageLockData(c->out_img_y, VPI_LOCK_READ, VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &stabilized_data) == VPI_SUCCESS) {
        const uint8_t* stabilized_src = (const uint8_t*)stabilized_data.buffer.pitch.planes[0].data;
        if (!output->data) {
            vpiImageUnlock(c->out_img_y);
            return PROC_STATUS_ERR_GENERAL;
        }
        uint8_t* output_dst = (uint8_t*)output->data;
        const int src_pitch = stabilized_data.buffer.pitch.planes[0].pitchBytes;
        copy_plane_host_to_vpi(output_dst, output->stride, stabilized_src, src_pitch, c->width, c->height);
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

    CHECK_STATUS(vpiSubmitConvertImageFormat(c->vpi_stream, VPI_BACKEND_VIC,
                                             c->cur_img_y, c->cur_img_y_bl, NULL));
    CHECK_STATUS(vpiStreamSync(c->vpi_stream));

    if (!c->has_prev_of) {
        c->has_prev_of = true;
        VPIImage tmp = c->prev_img_y_bl;
        c->prev_img_y_bl = c->cur_img_y_bl;
        c->cur_img_y_bl = tmp;
        return PROC_STATUS_OK;
    }

    CHECK_STATUS(vpiSubmitOpticalFlowDense(c->vpi_stream, VPI_BACKEND_OFA,
                                           c->ofa_payload,
                                           c->prev_img_y_bl, c->cur_img_y_bl,
                                           c->mv_img_bl));
    CHECK_STATUS(vpiSubmitConvertImageFormat(c->vpi_stream, VPI_BACKEND_VIC,
                                             c->mv_img_bl, c->mv_img_pl, NULL));
    CHECK_STATUS(vpiStreamSync(c->vpi_stream));

    st = nv_stab_estimate_motion(c);
    if (st != PROC_STATUS_OK) return st;

    // 4. Smooth motion trajectory
    nv_stab_smooth_motion(c);

    // 5. Apply stabilization
    st = nv_stab_apply_stabilization(c, input);
    if (st != PROC_STATUS_OK) return st;

    // 6. Update for next frame
    VPIImage tmp = c->prev_img_y_bl;
    c->prev_img_y_bl = c->cur_img_y_bl;
    c->cur_img_y_bl = tmp;

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

    if (c->cur_img_y_bl) vpiImageDestroy(c->cur_img_y_bl);
    if (c->prev_img_y_bl) vpiImageDestroy(c->prev_img_y_bl);
    if (c->mv_img_bl) vpiImageDestroy(c->mv_img_bl);
    if (c->mv_img_pl) vpiImageDestroy(c->mv_img_pl);

    // Destroy UV plane images
    if (c->cur_img_uv) vpiImageDestroy(c->cur_img_uv);
    if (c->prev_img_uv) vpiImageDestroy(c->prev_img_uv);
    if (c->out_img_uv) vpiImageDestroy(c->out_img_uv);

    if (c->ofa_payload) vpiPayloadDestroy(c->ofa_payload);

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
