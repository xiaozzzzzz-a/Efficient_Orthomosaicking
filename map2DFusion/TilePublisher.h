#ifndef MAP2DFUSION_TILEPUBLISHER_H
#define MAP2DFUSION_TILEPUBLISHER_H

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <opencv2/core.hpp>

class TilePublisher {
public:
    struct Options {
        std::string host = "127.0.0.1";
        int port = 8080;
        std::string taskId = "local";
        int level = 0;
        int timeoutMs = 2000;
        std::size_t maxQueueSize = 256;
    };

    struct TileJob {
        int level = 0;
        int x = 0;
        int y = 0;
        double timestamp = 0.0;
        double worldMinX = 0.0;
        double worldMinY = 0.0;
        double metersPerPixel = 0.0;
        cv::Mat image;
    };

    TilePublisher();
    explicit TilePublisher(const Options& options);
    ~TilePublisher();

    void start();
    void stop();

    void enqueue(TileJob job);
    bool publishNow(const TileJob& job, std::string* errorMessage = nullptr);

    std::size_t pendingCount() const;
    const Options& options() const { return options_; }

private:
    void workerLoop();
    bool publishEncodedTile(const TileJob& job,
                            const std::vector<uint8_t>& encoded,
                            std::string* errorMessage);
    std::string buildTilePath(const TileJob& job) const;

    Options options_;
    mutable std::mutex mutex_;
    std::condition_variable condition_;
    std::deque<TileJob> queue_;
    std::thread worker_;
    bool running_ = false;
    bool stopRequested_ = false;
};

#endif
