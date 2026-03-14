#include "recorder.hpp"

#include <fmt/chrono.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "math_tools.hpp"
#include "tools/logger.hpp"

namespace tools
{
Recorder::Recorder(double fps)
: init_(false),
  stop_thread_(false),
  fps_(fps),
  queue_(16)
{
  start_time_ = std::chrono::steady_clock::now();
  last_time_ = start_time_;

  auto folder_path = "records";
  auto file_name = fmt::format("{:%Y-%m-%d_%H-%M-%S}", std::chrono::system_clock::now());
  text_path_ = fmt::format("{}/{}.txt", folder_path, file_name);
  video_path_ = fmt::format("{}/{}.avi", folder_path, file_name);

  std::filesystem::create_directory(folder_path);
  tools::logger()->info("[Recorder] Video will be saved to {}", video_path_);
  tools::logger()->info("[Recorder] IMU log will be saved to {}", text_path_);
}

Recorder::~Recorder()
{
  stop_thread_ = true;
  // 退出时给队列中额外推入一个空帧，避免pop一直等待
  queue_.push({cv::Mat::zeros(0, 0, 0), {0, 0, 0, 0}, std::chrono::steady_clock::now()});
  if (saving_thread_.joinable()) saving_thread_.join();  // 等待视频保存线程结束

  if (!init_) return;
  text_writer_.close();
  video_writer_.release();
}

void Recorder::save_to_file()
{
  while (true) {
    FrameData frame;
    queue_.pop(frame);  // 从队列中取出帧数据
    if (stop_thread_ && frame.img.empty()) break;
    if (frame.img.empty()) {
      tools::logger()->debug("Recorder received empty img. Skip this frame.");
      continue;
    }
    // 写入视频文件
    video_writer_.write(frame.img);

    // 写入文本文件（输出顺序为wxyz）
    Eigen::Vector4d xyzw = frame.q.coeffs();
    auto since_begin = tools::delta_time(frame.timestamp, start_time_);
    text_writer_ << fmt::format(
      "{} {} {} {} {}\n", since_begin, xyzw[3], xyzw[0], xyzw[1], xyzw[2]);
  }
}

void Recorder::record(
  const cv::Mat & img, const Eigen::Quaterniond & q,
  const std::chrono::steady_clock::time_point & timestamp)
{
  if (img.empty()) return;
  if (!init_) init(img);

  auto since_last = tools::delta_time(timestamp, last_time_);
  if (since_last < 1.0 / fps_) return;

  last_time_ = timestamp;
  queue_.push({img.clone(), q.normalized(), timestamp});
}

void Recorder::init(const cv::Mat & img)
{
  text_writer_.open(text_path_);
  if (!text_writer_.is_open()) {
    throw std::runtime_error("Failed to open recorder text file: " + text_path_);
  }
  auto fourcc = cv::VideoWriter::fourcc('M', 'J', 'P', 'G');
  video_writer_ = cv::VideoWriter(video_path_, fourcc, fps_, img.size());
  if (!video_writer_.isOpened()) {
    throw std::runtime_error("Failed to open recorder video file: " + video_path_);
  }
  saving_thread_ = std::thread(&Recorder::save_to_file, this);  // 启动保存线程
  init_ = true;
}

}  // namespace tools
