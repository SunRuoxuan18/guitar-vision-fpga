/* Build from the pynq directory on the PYNQ board:
 * gcc -O3 -shared -fPIC -o libfast_jpeg.so native/fast_jpeg.c -ljpeg
 */
#include <limits.h>
#include <setjmp.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <jpeglib.h>
#include <jerror.h>

enum { FAST_JPEG_OK, FAST_JPEG_INVALID, FAST_JPEG_UNSUPPORTED, FAST_JPEG_LIBJPEG };

struct fast_jpeg_error {
    struct jpeg_error_mgr base;
    jmp_buf jump;
};

struct fast_jpeg_destination {
    struct jpeg_destination_mgr base;
    unsigned char *buffer;
    size_t capacity;
    size_t size;
};

struct fast_jpeg_context {
    struct jpeg_compress_struct cinfo;
    struct fast_jpeg_error error;
    struct fast_jpeg_destination dest;
};

struct fast_jpeg_decoder {
    struct jpeg_decompress_struct cinfo;
    struct fast_jpeg_error error;
};

static void fast_jpeg_error_exit(j_common_ptr cinfo) {
    struct fast_jpeg_error *error = (struct fast_jpeg_error *)cinfo->err;
    longjmp(error->jump, 1);
}

static void fast_jpeg_init_destination(j_compress_ptr cinfo) {
    struct fast_jpeg_destination *dest = (struct fast_jpeg_destination *)cinfo->dest;
    dest->buffer = (unsigned char *)malloc(4096);
    if (!dest->buffer) ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 10);
    dest->capacity = 4096;
    dest->size = 0;
    dest->base.next_output_byte = dest->buffer;
    dest->base.free_in_buffer = dest->capacity;
}

static boolean fast_jpeg_grow_destination(j_compress_ptr cinfo) {
    struct fast_jpeg_destination *dest = (struct fast_jpeg_destination *)cinfo->dest;
    unsigned char *buffer;
    size_t capacity;
    if (dest->capacity > SIZE_MAX / 2 || dest->capacity > ULONG_MAX / 2)
        ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 10);
    capacity = dest->capacity * 2;
    /* Keep ownership of the old allocation if realloc fails. */
    buffer = (unsigned char *)realloc(dest->buffer, capacity);
    if (!buffer) ERREXIT1(cinfo, JERR_OUT_OF_MEMORY, 10);
    dest->buffer = buffer;
    dest->base.next_output_byte = buffer + dest->capacity;
    dest->base.free_in_buffer = capacity - dest->capacity;
    dest->capacity = capacity;
    return TRUE;
}

static void fast_jpeg_term_destination(j_compress_ptr cinfo) {
    struct fast_jpeg_destination *dest = (struct fast_jpeg_destination *)cinfo->dest;
    dest->size = dest->capacity - dest->base.free_in_buffer;
}

const char *fast_jpeg_error_string(int status) {
    switch (status) {
    case FAST_JPEG_OK: return "success";
    case FAST_JPEG_INVALID: return "invalid JPEG input";
    case FAST_JPEG_UNSUPPORTED: return "libjpeg lacks JCS_EXT_BGR support";
    default: return "libjpeg JPEG failure";
    }
}

void fast_jpeg_free(unsigned char *buffer) {
    free(buffer);
}

int fast_jpeg_encode(const uint8_t *bgr, size_t input_size, int width, int height,
                     int stride, int quality, unsigned char **output,
                     unsigned long *output_size) {
    struct fast_jpeg_context *ctx;
    JSAMPROW row[1];

    if (!bgr || !output || !output_size || width <= 0 || height <= 0 ||
        width > 1280 || height > 720 || stride < width * 3 || quality < 1 ||
        quality > 100 || (size_t)stride > SIZE_MAX / (size_t)height ||
        input_size < (size_t)stride * (size_t)height) {
        return FAST_JPEG_INVALID;
    }
    *output = NULL;
    *output_size = 0;
#ifndef JCS_EXTENSIONS
    return FAST_JPEG_UNSUPPORTED;
#else
    /* Heap state remains defined after libjpeg longjmps into this function. */
    ctx = (struct fast_jpeg_context *)calloc(1, sizeof(*ctx));
    if (!ctx) return FAST_JPEG_LIBJPEG;
    ctx->cinfo.err = jpeg_std_error(&ctx->error.base);
    ctx->error.base.error_exit = fast_jpeg_error_exit;
    if (setjmp(ctx->error.jump)) {
        jpeg_destroy_compress(&ctx->cinfo);
        free(ctx->dest.buffer);
        free(ctx);
        return FAST_JPEG_LIBJPEG;
    }
    jpeg_create_compress(&ctx->cinfo);
    ctx->dest.base.init_destination = fast_jpeg_init_destination;
    ctx->dest.base.empty_output_buffer = fast_jpeg_grow_destination;
    ctx->dest.base.term_destination = fast_jpeg_term_destination;
    ctx->cinfo.dest = &ctx->dest.base;
    ctx->cinfo.image_width = (JDIMENSION)width;
    ctx->cinfo.image_height = (JDIMENSION)height;
    ctx->cinfo.input_components = 3;
    ctx->cinfo.in_color_space = JCS_EXT_BGR;
    jpeg_set_defaults(&ctx->cinfo);
    jpeg_set_quality(&ctx->cinfo, quality, TRUE);
    ctx->cinfo.comp_info[0].h_samp_factor = 2;
    ctx->cinfo.comp_info[0].v_samp_factor = 2;
    ctx->cinfo.comp_info[1].h_samp_factor = 1;
    ctx->cinfo.comp_info[1].v_samp_factor = 1;
    ctx->cinfo.comp_info[2].h_samp_factor = 1;
    ctx->cinfo.comp_info[2].v_samp_factor = 1;
    ctx->cinfo.dct_method = JDCT_IFAST;
    jpeg_start_compress(&ctx->cinfo, TRUE);
    while (ctx->cinfo.next_scanline < ctx->cinfo.image_height) {
        row[0] = (JSAMPROW)(bgr + (size_t)ctx->cinfo.next_scanline * (size_t)stride);
        jpeg_write_scanlines(&ctx->cinfo, row, 1);
    }
    jpeg_finish_compress(&ctx->cinfo);
    jpeg_destroy_compress(&ctx->cinfo);
    *output = ctx->dest.buffer;
    *output_size = (unsigned long)ctx->dest.size;
    free(ctx);
    return FAST_JPEG_OK;
#endif
}

/* Validate dimensions before allocating libjpeg's image workspace or writing
 * the caller's fixed RGBA buffer. Transport decoding stays entirely on PS. */
int fast_jpeg_decode_rgba(const uint8_t *jpeg, size_t input_size, uint8_t *rgba,
                          size_t output_size, int width, int height) {
    struct fast_jpeg_decoder *ctx;
    JSAMPROW row[1];
    if (!jpeg || !rgba || input_size < 4 || input_size > 1280u * 720u * 4u ||
        input_size > ULONG_MAX || width <= 0 || height <= 0 ||
        width > 1280 || height > 720 || output_size != (size_t)width * height * 4 ||
        jpeg[0] != 0xff || jpeg[1] != 0xd8)
        return FAST_JPEG_INVALID;
#ifndef JCS_EXTENSIONS
    return FAST_JPEG_UNSUPPORTED;
#else
    ctx = (struct fast_jpeg_decoder *)calloc(1, sizeof(*ctx));
    if (!ctx) return FAST_JPEG_LIBJPEG;
    ctx->cinfo.err = jpeg_std_error(&ctx->error.base);
    ctx->error.base.error_exit = fast_jpeg_error_exit;
    if (setjmp(ctx->error.jump)) {
        jpeg_destroy_decompress(&ctx->cinfo);
        free(ctx);
        return FAST_JPEG_LIBJPEG;
    }
    jpeg_create_decompress(&ctx->cinfo);
    ctx->cinfo.mem->max_memory_to_use = 16L * 1024L * 1024L;
    jpeg_mem_src(&ctx->cinfo, jpeg, (unsigned long)input_size);
    if (jpeg_read_header(&ctx->cinfo, TRUE) != JPEG_HEADER_OK ||
        ctx->cinfo.image_width != (JDIMENSION)width ||
        ctx->cinfo.image_height != (JDIMENSION)height ||
        (ctx->cinfo.num_components != 1 && ctx->cinfo.num_components != 3)) {
        jpeg_destroy_decompress(&ctx->cinfo);
        free(ctx);
        return FAST_JPEG_INVALID;
    }
    ctx->cinfo.out_color_space = JCS_EXT_RGBA;
    ctx->cinfo.dct_method = JDCT_ISLOW;
    jpeg_start_decompress(&ctx->cinfo);
    if (ctx->cinfo.output_components != 4 ||
        ctx->cinfo.output_width != (JDIMENSION)width ||
        ctx->cinfo.output_height != (JDIMENSION)height) {
        jpeg_destroy_decompress(&ctx->cinfo);
        free(ctx);
        return FAST_JPEG_INVALID;
    }
    while (ctx->cinfo.output_scanline < ctx->cinfo.output_height) {
        row[0] = rgba + (size_t)ctx->cinfo.output_scanline * width * 4;
        if (jpeg_read_scanlines(&ctx->cinfo, row, 1) != 1)
            ERREXIT(&ctx->cinfo, JERR_INPUT_EOF);
    }
    jpeg_finish_decompress(&ctx->cinfo);
    int status = ctx->error.base.num_warnings ? FAST_JPEG_INVALID : FAST_JPEG_OK;
    jpeg_destroy_decompress(&ctx->cinfo);
    free(ctx);
    return status;
#endif
}
