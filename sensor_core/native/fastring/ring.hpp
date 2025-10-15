#pragma once
#include <cstdint>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

struct RingHeader {
    std::atomic<uint64_t> write_idx;
    size_t capacity;
    size_t frame_bytes;
};

struct ShmRing {
    int fd = -1;
    size_t capacity = 0;
    size_t frame_bytes = 0;
    size_t total_bytes = 0;
    uint8_t* base = nullptr;
    RingHeader* hdr = nullptr;
    uint8_t* data = nullptr;

    static ShmRing create(const char* name, size_t capacity, size_t frame_bytes) {
        ShmRing r;
        r.capacity = capacity;
        r.frame_bytes = frame_bytes;
        r.total_bytes = sizeof(RingHeader) + capacity * frame_bytes;
        int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
        if (fd < 0) throw std::runtime_error("shm_open create failed");
        if (ftruncate(fd, r.total_bytes) != 0) { close(fd); throw std::runtime_error("ftruncate failed"); }
        void* p = mmap(nullptr, r.total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
        r.fd = fd; r.base = (uint8_t*)p; r.hdr = (RingHeader*)r.base; r.data = r.base + sizeof(RingHeader);
        r.hdr->write_idx.store(0, std::memory_order_relaxed);
        r.hdr->capacity = capacity; r.hdr->frame_bytes = frame_bytes;
        return r;
    }
    static ShmRing open(const char* name, size_t capacity, size_t frame_bytes) {
        ShmRing r;
        r.capacity = capacity; r.frame_bytes = frame_bytes;
        r.total_bytes = sizeof(RingHeader) + capacity * frame_bytes;
        int fd = shm_open(name, O_RDWR, 0600);
        if (fd < 0) throw std::runtime_error("shm_open open failed");
        void* p = mmap(nullptr, r.total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) { close(fd); throw std::runtime_error("mmap failed"); }
        r.fd = fd; r.base = (uint8_t*)p; r.hdr = (RingHeader*)r.base; r.data = r.base + sizeof(RingHeader);
        return r;
    }
    void publish(const void* frames, size_t nframes) {
        const uint8_t* src = (const uint8_t*)frames;
        uint64_t idx = hdr->write_idx.load(std::memory_order_relaxed);
        for (size_t i = 0; i < nframes; ++i) {
            size_t slot = (size_t)((idx + i) % capacity);
            std::memcpy(data + slot * frame_bytes, src + i * frame_bytes, frame_bytes);
        }
        hdr->write_idx.store(idx + nframes, std::memory_order_release);
    }
};
