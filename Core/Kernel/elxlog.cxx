/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/

// Its own header
#include "elxlog.h"

// Rename the spdlog namespace, to avoid naming conflicts.
#define spdlog elx_spdlog

// The logging library from https://github.com/gabime/spdlog
#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_sinks.h>

#include <cassert>
#include <memory> // For make_shared.

namespace elastix
{
namespace
{

class log_data
{
public:
  auto &
  get_log_file_logger()
  {
    return m_loggers.log_file_logger;
  }

  auto &
  get_stdout_logger()
  {
    return m_loggers.stdout_logger;
  }

  auto &
  get_multi_logger()
  {
    return m_loggers.multi_logger;
  }

  void
  reset()
  {
    m_loggers = {};
  }

private:
  struct loggers
  {
    spdlog::logger log_file_logger{ "elastix file logger" };
    spdlog::logger stdout_logger{ "elastix stdout logger" };
    spdlog::logger multi_logger{ "elastix multi logger" };
  };

  loggers m_loggers;
};


auto &
get_log_data()
{
  static log_data static_log_data{};

  return static_log_data;
}

auto &
get_multi_logger()
{
  return get_log_data().get_multi_logger();
}

std::string
get_string_from_stream(const std::ostream & stream)
{
  const auto * const outputstringstream = dynamic_cast<const std::ostringstream *>(&stream);

  if (outputstringstream)
  {
    return outputstringstream->str();
  }
  assert(!"Failed to convert std::ostream pointer to std::ostringstream pointer!");
  log::error("Failed to convert std::ostream pointer to std::ostringstream pointer");
  return {};
}


void
setup_implementation(const std::string & log_filename, const bool do_log_to_file, const bool do_log_to_stdout)
{
  std::vector<spdlog::sink_ptr> log_file_sink;
  std::vector<spdlog::sink_ptr> stdout_sink;
  std::vector<spdlog::sink_ptr> all_sinks;

  if (do_log_to_file)
  {
    // Throws an exception when the log file cannot be opened.
    auto sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_filename, true);
    sink->set_pattern("%v");

    log_file_sink = { sink };
    all_sinks.push_back(sink);
  }

  if (do_log_to_stdout)
  {
    auto sink = std::make_shared<spdlog::sinks::stdout_sink_mt>();
    sink->set_pattern("%v");

    stdout_sink = { sink };
    all_sinks.push_back(sink);
  }

  auto & data = get_log_data();

  // From here, the code is exception free, so all sinks will be set properly.
  data.get_log_file_logger().sinks() = std::move(log_file_sink);
  data.get_stdout_logger().sinks() = std::move(stdout_sink);
  data.get_multi_logger().sinks() = std::move(all_sinks);
}

} // namespace


bool
log::setup(const std::string & log_filename, const bool do_log_to_file, const bool do_log_to_stdout)
{
  try
  {
    setup_implementation(log_filename, do_log_to_file, do_log_to_stdout);
    return true;
  }
  catch (const std::exception &)
  {
    return false;
  }
}


log::guard::guard() = default;

log::guard::guard(const std::string & log_filename, const bool do_log_to_file, const bool do_log_to_stdout)
{
  setup_implementation(log_filename, do_log_to_file, do_log_to_stdout);
}

log::guard::~guard()
{
  get_log_data().reset();
}


void
log::info(const std::string & message)
{
  get_multi_logger().info(message);
}

void
log::warn(const std::string & message)
{
  get_multi_logger().warn(message);
}

void
log::error(const std::string & message)
{
  get_multi_logger().error(message);
}


void
log::info(const std::ostream & stream)
{
  get_multi_logger().info(get_string_from_stream(stream));
}

void
log::warn(const std::ostream & stream)
{
  get_multi_logger().warn(get_string_from_stream(stream));
}

void
log::error(const std::ostream & stream)
{
  get_multi_logger().error(get_string_from_stream(stream));
}

void
log::to_log_file(const std::string & message)
{
  get_log_data().get_log_file_logger().info(message);
}

void
log::to_log_file(const std::ostream & stream)
{
  log::to_log_file(get_string_from_stream(stream));
}

void
log::to_stdout(const std::string & message)
{
  get_log_data().get_stdout_logger().info(message);
}

void
log::to_stdout(const std::ostream & stream)
{
  log::to_stdout(get_string_from_stream(stream));
}

} // end namespace elastix
