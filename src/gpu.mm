// gpu.mm — Metal harness implementation (Objective-C++).
// Unified memory: every Buf is an MTLBuffer with a host-visible pointer kept in a
// registry so dispatch() can bind raw pointers back to their MTLBuffer objects.
#import <Metal/Metal.h>
#include "gpu.h"
#include <chrono>
#include <cstring>
#include <map>
#include <mutex>
#include <unordered_map>

namespace tn::gpu {

struct RegEntry {
    uintptr_t base;
    size_t length;
    id<MTLBuffer> buf;
};

struct Impl {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLLibrary> library = nil;
    id<MTLCommandBuffer> cmd = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines;
    std::map<uintptr_t, RegEntry> bufreg;
    std::mutex mutex;
    bool encode_failed = false;
};

Device::~Device() { delete impl_; }

bool Device::init(const std::string & metallib_path) {
    impl_ = new Impl();
    impl_->device = MTLCreateSystemDefaultDevice();
    if (!impl_->device) { fprintf(stderr, "gpu: no Metal device\n"); return false; }
    impl_->queue = [impl_->device newCommandQueue];
    NSError * err = nil;
    NSString * libPath = [NSString stringWithUTF8String:metallib_path.c_str()];
    NSURL * libURL = [NSURL fileURLWithPath:libPath];
    id<MTLLibrary> lib = [impl_->device newLibraryWithURL:libURL error:&err];
    if (!lib) {
        fprintf(stderr, "gpu: library load failed (%s): %s\n", metallib_path.c_str(),
                err.localizedDescription.UTF8String);
        return false;
    }
    impl_->library = lib;
    return true;
}

Buf Device::alloc(size_t bytes) {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    Buf b;
    if (bytes == 0) bytes = 16;
    id<MTLBuffer> buf = [impl_->device newBufferWithLength:bytes
        options:MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked];
    if (!buf) return b;
    uintptr_t addr = (uintptr_t) buf.contents;
    impl_->bufreg[addr] = { addr, bytes, buf };
    b.p = buf.contents;
    b.bytes = bytes;
    return b;
}

Buf Device::wrap(const void * p, size_t bytes) {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    Buf b;
    uintptr_t addr = (uintptr_t) p;
    auto it = impl_->bufreg.upper_bound(addr);
    if (it != impl_->bufreg.begin()) {
        --it;
        if (addr >= it->second.base && addr < it->second.base + it->second.length) {
            b.p = (void *) p;
            b.bytes = bytes;
            return b;
        }
    }
    id<MTLBuffer> buf = [impl_->device newBufferWithBytesNoCopy:(void *)p length:bytes
        options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined
        deallocator:^(void *, NSUInteger) {}];
    if (!buf) { fprintf(stderr, "gpu: wrap failed for %p (%zu B)\n", p, bytes); return b; }
    impl_->bufreg[addr] = { addr, bytes, buf };
    b.p = (void *) p;
    b.bytes = bytes;
    return b;
}

void Device::begin() {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    if (impl_->enc) { [impl_->enc endEncoding]; impl_->enc = nil; }
    if (impl_->cmd) { [impl_->cmd commit]; [impl_->cmd waitUntilCompleted]; }
    impl_->cmd = [impl_->queue commandBufferWithUnretainedReferences];
    impl_->enc = [impl_->cmd computeCommandEncoder];
    impl_->encode_failed = false;
}

void Device::dispatch(const std::string & kernel,
                      uint32_t gx, uint32_t gy, uint32_t gz,
                      uint32_t tx, uint32_t ty, uint32_t tz,
                      const std::vector<const void *> & buffers,
                      const void * scalars, size_t scalar_bytes) {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    if (!impl_->enc) {
        impl_->cmd = [impl_->queue commandBufferWithUnretainedReferences];
        impl_->enc = [impl_->cmd computeCommandEncoder];
    }
    id<MTLComputePipelineState> ps;
    auto it = impl_->pipelines.find(kernel);
    if (it == impl_->pipelines.end()) {
        NSString * name = [NSString stringWithUTF8String:kernel.c_str()];
        id<MTLFunction> fn = [impl_->library newFunctionWithName:name];
        if (!fn) { fprintf(stderr, "gpu: kernel %s not found\n", kernel.c_str()); impl_->encode_failed = true; return; }
        NSError * err = nil;
        ps = [impl_->device newComputePipelineStateWithFunction:fn error:&err];
        if (!ps) { fprintf(stderr, "gpu: pipeline %s failed: %s\n", kernel.c_str(),
                           err.localizedDescription.UTF8String); impl_->encode_failed = true; return; }
        impl_->pipelines[kernel] = ps;
    } else {
        ps = it->second;
    }
    [impl_->enc setComputePipelineState:ps];
    for (size_t i = 0; i < buffers.size() && i < 29; ++i) {
        void * bp = const_cast<void *>(buffers[i]);
        if (bp == nullptr) {
            [impl_->enc setBuffer:nil offset:0 atIndex:(uint32_t) i];
            continue;
        }
        uintptr_t addr = (uintptr_t) bp;
        auto bi = impl_->bufreg.upper_bound(addr);
        if (bi != impl_->bufreg.begin()) {
            --bi;
            const auto & ent = bi->second;
            if (addr >= ent.base && addr < ent.base + ent.length) {
                NSUInteger offset = addr - ent.base;
                [impl_->enc setBuffer:ent.buf offset:offset atIndex:(uint32_t) i];
                continue;
            }
        }
        fprintf(stderr, "gpu: unregistered buffer %p for %s\n", bp, kernel.c_str());
        impl_->encode_failed = true;
        return;
    }
    if (scalars && scalar_bytes)
        [impl_->enc setBytes:scalars length:scalar_bytes atIndex:(uint32_t) buffers.size()];
    MTLSize tg = MTLSizeMake(tx, ty, tz);
    if (tg.width * tg.height * tg.depth > ps.maxTotalThreadsPerThreadgroup) {
        fprintf(stderr, "gpu: tg %ux%ux%u too large for %s\n", tx, ty, tz, kernel.c_str());
        impl_->encode_failed = true;
        return;
    }
    MTLSize groups = MTLSizeMake(gx, gy, gz);
    [impl_->enc dispatchThreadgroups:groups threadsPerThreadgroup:tg];
}

void Device::commit() {
    std::lock_guard<std::mutex> lk(impl_->mutex);
    if (impl_->enc) { [impl_->enc endEncoding]; impl_->enc = nil; }
    if (!impl_->cmd) return;
    auto start = std::chrono::steady_clock::now();
    id<MTLCommandBuffer> cmd = impl_->cmd;
    impl_->cmd = nil;
    [cmd commit];
    [cmd waitUntilCompleted];
    auto end = std::chrono::steady_clock::now();
    last_gpu_ms_ = std::chrono::duration<double, std::milli>(end - start).count();
    if (cmd.error)
        fprintf(stderr, "gpu: commit error: %s\n", cmd.error.localizedDescription.UTF8String);
}

} // namespace tn::gpu
