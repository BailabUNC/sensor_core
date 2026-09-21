#pragma once
#include <cstdint>
#include <atomic>
#include <cstring>
#include <stdexcept>
#include <string>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <fcntl.h>
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <unistd.h>
#endif

struct RingHeader {
    std::atomic<uint64_t> write_idx;
    size_t capacity;
    size_t frame_bytes;
};

// Shared-memory layout: RingHeader, then one uint64 timestamp per slot, then the frame slots.

struct ShmRing {
#ifdef _WIN32
    HANDLE hMap = NULL;
#else
    int fd = -1;
#endif
    size_t capacity = 0;
    size_t frame_bytes = 0;
    size_t total_bytes = 0;
    uint8_t* base = nullptr;
    RingHeader* hdr = nullptr;
    uint64_t* ts = nullptr;   // per-slot timestamps, in whatever clock the publisher uses
    uint8_t* data = nullptr;

    ShmRing() = default;

    ShmRing(const ShmRing&) = delete;
    ShmRing& operator=(const ShmRing&) = delete;

    // Move constructor
    ShmRing(ShmRing&& other) noexcept {
        move_from(std::move(other));
    }

    // Move assignment
    ShmRing& operator=(ShmRing&& other) noexcept {
        if (this != &other) {
            cleanup();
            move_from(std::move(other));
        }
        return *this;
    }

    ~ShmRing() {
        cleanup();
    }

    static ShmRing create(const char* name, size_t capacity, size_t frame_bytes) {
        ShmRing r;
        r.capacity = capacity;
        r.frame_bytes = frame_bytes;
        r.total_bytes = sizeof(RingHeader) + capacity * (sizeof(uint64_t) + frame_bytes);

#ifdef _WIN32
        LARGE_INTEGER li;
        li.QuadPart = static_cast<LONGLONG>(r.total_bytes);
        HANDLE hMap = CreateFileMappingA(
            INVALID_HANDLE_VALUE,
            NULL,
            PAGE_READWRITE,
            li.HighPart,
            li.LowPart,
            name
        );
        if (!hMap)
            throw std::runtime_error("CreateFileMapping failed");

        void* p = MapViewOfFile(
            hMap,
            FILE_MAP_ALL_ACCESS,
            0, 0,
            r.total_bytes
        );
        if (!p) {
            CloseHandle(hMap);
            throw std::runtime_error("MapViewOfFile failed");
        }

        r.hMap = hMap;
        r.base = static_cast<uint8_t*>(p);
#else
        int fd = shm_open(name, O_CREAT | O_RDWR, 0600);
        if (fd < 0)
            throw std::runtime_error("shm_open create failed");

        if (ftruncate(fd, static_cast<off_t>(r.total_bytes)) != 0) {
            close(fd);
            throw std::runtime_error("ftruncate failed");
        }

        void* p = mmap(nullptr, r.total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed");
        }

        r.fd = fd;
        r.base = static_cast<uint8_t*>(p);
#endif

        r.hdr  = reinterpret_cast<RingHeader*>(r.base);
        r.ts   = reinterpret_cast<uint64_t*>(r.base + sizeof(RingHeader));
        r.data = r.base + sizeof(RingHeader) + capacity * sizeof(uint64_t);
        r.hdr->write_idx.store(0, std::memory_order_relaxed);
        r.hdr->capacity    = capacity;
        r.hdr->frame_bytes = frame_bytes;
        return r;
    }

    static ShmRing open(const char* name, size_t capacity, size_t frame_bytes) {
        ShmRing r;
        r.capacity = capacity;
        r.frame_bytes = frame_bytes;
        r.total_bytes = sizeof(RingHeader) + capacity * (sizeof(uint64_t) + frame_bytes);

#ifdef _WIN32
        HANDLE hMap = OpenFileMappingA(
            FILE_MAP_ALL_ACCESS,
            FALSE,
            name
        );
        if (!hMap)
            throw std::runtime_error("OpenFileMapping failed");

        void* p = MapViewOfFile(
            hMap,
            FILE_MAP_ALL_ACCESS,
            0, 0,
            r.total_bytes
        );
        if (!p) {
            CloseHandle(hMap);
            throw std::runtime_error("MapViewOfFile failed");
        }

        r.hMap = hMap;
        r.base = static_cast<uint8_t*>(p);
#else
        int fd = shm_open(name, O_RDWR, 0600);
        if (fd < 0)
            throw std::runtime_error("shm_open open failed");

        void* p = mmap(nullptr, r.total_bytes, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            close(fd);
            throw std::runtime_error("mmap failed");
        }

        r.fd = fd;
        r.base = static_cast<uint8_t*>(p);
#endif

        r.hdr  = reinterpret_cast<RingHeader*>(r.base);
        r.ts   = reinterpret_cast<uint64_t*>(r.base + sizeof(RingHeader));
        r.data = r.base + sizeof(RingHeader) + capacity * sizeof(uint64_t);
        if (r.hdr->capacity != capacity || r.hdr->frame_bytes != frame_bytes) {
            throw std::runtime_error(
                std::string("ring layout mismatch: '") + name + "' was created with capacity " +
                std::to_string(r.hdr->capacity) + " and frame_bytes " + std::to_string(r.hdr->frame_bytes) +
                ", but opened with capacity " + std::to_string(capacity) +
                " and frame_bytes " + std::to_string(frame_bytes));
        }
        return r;
    }

    // Remove the ring's name so no new process can open it. Processes that already mapped the
    // ring keep their mapping until they release it. Windows frees the mapping automatically
    // once its last handle closes, so there is nothing to remove there.
    static void unlink(const char* name) {
#ifndef _WIN32
        shm_unlink(name);
#endif
    }

    // Copy nframes frames into the ring, each stamped with ts_ns. Each frame becomes visible to readers as
    // soon as it is complete, so at most one slot (the oldest frame's) is ever being rewritten.
    void publish(const void* frames, size_t nframes, uint64_t ts_ns) {
        const uint8_t* src = static_cast<const uint8_t*>(frames);
        uint64_t idx = hdr->write_idx.load(std::memory_order_relaxed);

        for (size_t i = 0; i < nframes; ++i) {
            size_t slot = static_cast<size_t>((idx + i) % capacity);
            ts[slot] = ts_ns;
            std::memcpy(data + slot * frame_bytes, src + i * frame_bytes, frame_bytes);
            hdr->write_idx.store(idx + i + 1, std::memory_order_release);
        }
    }

private:
    void cleanup() {
#ifdef _WIN32
        if (base) {
            UnmapViewOfFile(base);
        }
        if (hMap) {
            CloseHandle(hMap);
        }
        hMap = NULL;
#else
        if (base && total_bytes > 0) {
            munmap(base, total_bytes);
        }
        if (fd >= 0) {
            close(fd);
        }
        fd = -1;
#endif
        base = nullptr;
        hdr = nullptr;
        ts = nullptr;
        data = nullptr;
        capacity = 0;
        frame_bytes = 0;
        total_bytes = 0;
    }

    void move_from(ShmRing&& other) noexcept {
#ifdef _WIN32
        hMap = other.hMap;
        other.hMap = NULL;
#else
        fd = other.fd;
        other.fd = -1;
#endif
        capacity    = other.capacity;
        frame_bytes = other.frame_bytes;
        total_bytes = other.total_bytes;
        base        = other.base;
        hdr         = other.hdr;
        ts          = other.ts;
        data        = other.data;

        other.capacity = 0;
        other.frame_bytes = 0;
        other.total_bytes = 0;
        other.base = nullptr;
        other.hdr  = nullptr;
        other.ts   = nullptr;
        other.data = nullptr;
    }
};
