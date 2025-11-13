#pragma once
#include <cstdint>
#include <atomic>
#include <cstring>
#include <stdexcept>

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
        r.total_bytes = sizeof(RingHeader) + capacity * frame_bytes;

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
        r.data = r.base + sizeof(RingHeader);
        r.hdr->write_idx.store(0, std::memory_order_relaxed);
        r.hdr->capacity    = capacity;
        r.hdr->frame_bytes = frame_bytes;
        return r;
    }

    static ShmRing open(const char* name, size_t capacity, size_t frame_bytes) {
        ShmRing r;
        r.capacity = capacity;
        r.frame_bytes = frame_bytes;
        r.total_bytes = sizeof(RingHeader) + capacity * frame_bytes;

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
        r.data = r.base + sizeof(RingHeader);
        return r; 
    }

    void publish(const void* frames, size_t nframes) {
        const uint8_t* src = static_cast<const uint8_t*>(frames);
        uint64_t idx = hdr->write_idx.load(std::memory_order_relaxed);

        for (size_t i = 0; i < nframes; ++i) {
            size_t slot = static_cast<size_t>((idx + i) % capacity);
            std::memcpy(data + slot * frame_bytes, src + i * frame_bytes, frame_bytes);
        }

        hdr->write_idx.store(idx + nframes, std::memory_order_release);
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
        data        = other.data;

        other.capacity = 0;
        other.frame_bytes = 0;
        other.total_bytes = 0;
        other.base = nullptr;
        other.hdr  = nullptr;
        other.data = nullptr;
    }
};
