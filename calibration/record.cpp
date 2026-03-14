#include <fmt/core.h>

#include <chrono>
#include <opencv2/opencv.hpp>

#include "io/camera.hpp"
#include "io/gkdcontrol.hpp"
#include "tools/exiter.hpp"
#include "tools/img_tools.hpp"
#include "tools/logger.hpp"
#include "tools/math_tools.hpp"
#include "tools/recorder.hpp"

using namespace std::chrono_literals;

const std::string keys =
  "{help h usage ? |                           | 输出命令行参数说明}"
  "{@config-path c  | configs/gkdinfantry.yaml | yaml配置文件路径 }"
  "{fps f           | 30                        | 录制帧率         }"
  "{display d       |                           | 显示预览         }"
  "{scale s         | 0.5                       | 预览缩放比例     }";

int main(int argc, char * argv[])
{
  cv::CommandLineParser cli(argc, argv, keys);
  if (cli.has("help")) {
    cli.printMessage();
    return 0;
  }

  auto config_path = cli.get<std::string>(0);
  auto fps = cli.get<double>("fps");
  auto display = cli.has("display");
  auto scale = cli.get<double>("scale");
  if (config_path.empty()) {
    cli.printMessage();
    return 1;
  }
  if (fps <= 0) {
    tools::logger()->error("[record] fps must be positive, got {}", fps);
    return 1;
  }
  if (scale <= 0) {
    tools::logger()->error("[record] scale must be positive, got {}", scale);
    return 1;
  }

  tools::Exiter exiter;
  tools::Recorder recorder(fps);
  io::GKDControl gkdcontrol(config_path);
  io::Camera camera(config_path);

  tools::logger()->info("[record] Recording started. Press Ctrl+C to stop.");
  if (display) tools::logger()->info("[record] Preview enabled. Press q in the preview window to stop.");

  cv::Mat img;
  std::chrono::steady_clock::time_point timestamp;
  int frame_count = 0;
  while (!exiter.exit()) {
    camera.read(img, timestamp);
    if (img.empty()) continue;

    Eigen::Quaterniond q = gkdcontrol.imu_at(timestamp - 1ms);
    recorder.record(img, q, timestamp);

    if (display) {
      auto preview = img.clone();
      Eigen::Vector3d ypr = tools::eulers(q, 2, 1, 0) * 57.3;
      tools::draw_text(preview, fmt::format("frame {}", frame_count), {40, 40}, {0, 255, 255});
      tools::draw_text(preview, fmt::format("yaw   {:.2f}", ypr[0]), {40, 80}, {0, 0, 255});
      tools::draw_text(preview, fmt::format("pitch {:.2f}", ypr[1]), {40, 120}, {0, 0, 255});
      tools::draw_text(preview, fmt::format("roll  {:.2f}", ypr[2]), {40, 160}, {0, 0, 255});
      if (scale != 1.0) cv::resize(preview, preview, {}, scale, scale);
      cv::imshow("record", preview);
      if (cv::waitKey(1) == 'q') break;
    }

    frame_count++;
  }

  cv::destroyAllWindows();
  tools::logger()->info("[record] Recording stopped.");
  return 0;
}
