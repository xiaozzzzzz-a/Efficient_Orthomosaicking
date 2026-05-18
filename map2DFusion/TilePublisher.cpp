#include "TilePublisher.h"

#include <chrono>
#include <iostream>
#include <sstream>
#include <utility>

#include <opencv2/imgcodecs.hpp>

#include "httplib.h"

namespace {

double NowSeconds() {
    using clock = std::chrono::system_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

}  // namespace

TilePublisher::TilePublisher() : options_(Options()) {}

TilePublisher::TilePublisher(const Options& options) : options_(options) {}

TilePublisher::~TilePublisher() {
    stop();
}

void TilePublisher::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (running_) {
        return;
    }
    stopRequested_ = false;
    running_ = true;
    worker_ = std::thread(&TilePublisher::workerLoop, this);
}

void TilePublisher::stop() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!running_) {
            return;
        }
        stopRequested_ = true;
    }
    condition_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = false;
}

void TilePublisher::enqueue(TileJob job) {
    if (job.image.empty()) {
        return;
    }
    if (job.timestamp <= 0.0) {
        job.timestamp = NowSeconds();
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.size() >= options_.maxQueueSize) {
            queue_.pop_front();
        }
        queue_.push_back(std::move(job));
    }
    condition_.notify_one();
}

bool TilePublisher::publishNow(const TileJob& job, std::string* errorMessage) {
    if (job.image.empty()) {
        if (errorMessage) {
            *errorMessage = "tile image is empty";
        }
        return false;
    }

    std::vector<uint8_t> encoded;
    if (!cv::imencode(".png", job.image, encoded)) {
        if (errorMessage) {
            *errorMessage = "failed to encode tile image as PNG";
        }
        return false;
    }

    return publishEncodedTile(job, encoded, errorMessage);
}

std::size_t TilePublisher::pendingCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return queue_.size();
}

void TilePublisher::workerLoop() {
    while (true) {
        TileJob job;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            condition_.wait(lock, [this]() { return stopRequested_ || !queue_.empty(); });
            if (stopRequested_ && queue_.empty()) {
                break;
            }
            job = std::move(queue_.front());
            queue_.pop_front();
        }

        std::string errorMessage;
        if (!publishNow(job, &errorMessage)) {
            // std::cerr << "[TilePublisher] Failed to publish tile (" << job.level << ", "
            //           << job.x << ", " << job.y << "): " << errorMessage << std::endl;
        }
    }
}

bool TilePublisher::publishEncodedTile(const TileJob& job,
                                       const std::vector<uint8_t>& encoded,
                                       std::string* errorMessage) {
    httplib::Client client(options_.host.c_str(), options_.port,
                           std::max(1, options_.timeoutMs / 1000));
    const time_t timeoutSeconds = std::max(1, options_.timeoutMs / 1000);
    const time_t timeoutUsec = static_cast<time_t>((options_.timeoutMs % 1000) * 1000);
    client.set_read_timeout(timeoutSeconds, timeoutUsec);

    httplib::Headers headers = {
        {"Content-Type", "image/png"},
        {"X-Local-Level", std::to_string(job.level)},
        {"X-Local-Tile-X", std::to_string(job.x)},
        {"X-Local-Tile-Y", std::to_string(job.y)},
        {"X-Local-World-Min-X", std::to_string(job.worldMinX)},
        {"X-Local-World-Min-Y", std::to_string(job.worldMinY)},
        {"X-Local-Meters-Per-Pixel", std::to_string(job.metersPerPixel)},
    };

    const std::string path = buildTilePath(job);
    const std::string body(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    auto response = client.Post(path.c_str(), headers, body, "image/png");
    if (!response) {
        if (errorMessage) {
            *errorMessage = "HTTP request failed";
        }
        return false;
    }
    if (response->status < 200 || response->status >= 300) {
        if (errorMessage) {
            std::ostringstream oss;
            oss << "HTTP " << response->status << ": " << response->body;
            *errorMessage = oss.str();
        }
        return false;
    }
    return true;
}

std::string TilePublisher::buildTilePath(const TileJob& job) const {
    std::ostringstream oss;
    oss << "/dom/" << options_.taskId << "/tiles/"
        << job.level << "-" << job.x << "-" << job.y
        << "?timestamp=" << job.timestamp;
    return oss.str();
}
